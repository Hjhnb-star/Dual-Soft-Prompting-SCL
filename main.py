import os
import gc
import json
import torch
import random
import argparse
import numpy as np

# 环境变量设置
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from modelscope import AutoModelForSequenceClassification, snapshot_download
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType, LoraConfig, PromptEncoderConfig, PrefixTuningConfig, IA3Config

# 从我们拆分的模块中导入（假定你已经将旧代码的数据处理和Trainer提取到了这两个文件中）
from data_utils import get_data, DATASETS_CONFIG
from trainer_utils import DualPromptSCLTrainer, compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(args.seed)
    
    # 强制缓存路径
    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    
    print(f"🔽 Loading Model: {args.model_name}")
    model_cache_dir = snapshot_download("AI-ModelScope/roberta-large", cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)

    print(f"📦 Loading Dataset: {args.dataset} ({args.num_shots}-shot)")
    train_ds, eval_ds, n_classes = get_data(args.dataset, tokenizer, args.num_shots, args.cache_dir, args.seed)

    print(f"⚡ Initializing Method: {args.method}")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_cache_dir, num_labels=n_classes)

    # 动态配置 PEFT
    if args.method == "LoRA":
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, target_modules=["query", "value"])
    elif args.method == "IA3-Adapter":
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["query", "value", "dense"])
    elif args.method == "Prefix-Tuning":
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
    elif args.method in ["Vanilla-PT", "Ours-DualPromptSCL"]:
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=20,
            prompt_tuning_init_text="Classify if the text contains toxic or offensive language:",
            tokenizer_name_or_path=model_cache_dir,
        )
    elif args.method == "Ours-DeepDualPromptSCL":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS, num_virtual_tokens=20,
            encoder_reparameterization_type="LSTM", encoder_hidden_size=128
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    model = get_peft_model(base_model, peft_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"out_{args.dataset}_{args.method}_{args.num_shots}shot"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", 
        greater_is_better=True, 
        save_total_limit=1, 
        logging_steps=10,
        fp16=True,
        report_to="none"
    )

    # Trainer 路由
    if "Ours" in args.method:
        model.register_parameter(
            'global_prototypes',
            torch.nn.Parameter(torch.randn(n_classes, base_model.config.hidden_size).cuda())
        )
        torch.nn.init.xavier_uniform_(model.global_prototypes)
        
        trainer = DualPromptSCLTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer),
            scl_alpha=args.scl_alpha, cls_index=20, # 匹配 Prompt Tokens 的偏移
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )
    else:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )

    # 训练与评估
    trainer.train()
    res = trainer.evaluate()
    print(f"\n✅ {args.method} on {args.dataset} ({args.num_shots}-shot) - Eval F1: {res['eval_f1']:.4f}")

    # 保存单次实验结果
    res_path = os.path.join(args.output_dir, f"results_{args.dataset}_{args.num_shots}shot.json")
    # 这里可以添加将 res['eval_f1'] 写入 JSON 的逻辑

    # 显存清理
    del model, base_model, trainer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Soft-Prompting SCL for Extreme Few-Shot Detection")
    parser.add_argument("--dataset", type=str, default="jigsaw", choices=["jigsaw", "olid"])
    parser.add_argument("--method", type=str, default="Ours-DualPromptSCL", choices=["LoRA", "IA3-Adapter", "Prefix-Tuning", "Vanilla-PT", "Ours-DualPromptSCL", "Ours-DeepDualPromptSCL"])
    parser.add_argument("--num_shots", type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--scl_alpha", type=float, default=0.2, help="Weight for Prototype-Centered SCL loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    args = parser.parse_args()
    main(args)