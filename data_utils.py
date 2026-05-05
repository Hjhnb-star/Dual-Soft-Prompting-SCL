import os
from datasets import load_dataset, concatenate_datasets

# ===================== 数据集配置 =====================
DATASETS_CONFIG = {
    "jigsaw": {
        "path": "csv",
        "data_files": {"train": "https://hf-mirror.com/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv"},
        "text": "comment_text",
        "label": "toxic"
    },
    "olid": {
        "path": "parquet", 
        "data_files": {
            "train": "./data/tweet_eval/offensive/train-00000-of-00001.parquet" 
        },
        "text": "text",
        "label": "label"
    }
}

def get_data(dataset_key, tokenizer, num_shots, cache_dir, seed=42):
    """
    加载并处理极端少样本数据集
    """
    cfg = DATASETS_CONFIG[dataset_key]
    
    # 根据数据集类型加载
    if dataset_key == "olid":
        ds = load_dataset(cfg["path"], data_files=cfg["data_files"], split="train", cache_dir=cache_dir)
    else:
        ds = load_dataset(
            cfg["path"],
            data_files=cfg["data_files"],
            split="train",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    label_col, text_col = cfg["label"], cfg["text"]
    unique_labels = sorted(list(set(ds[label_col])))
    train_sets, eval_sets = [], []
    
    # 按类别进行 Few-shot 采样
    for label in unique_labels:
        cls_ds = ds.filter(lambda x: x[label_col] == label).shuffle(seed=seed)
        train_sets.append(cls_ds.select(range(min(num_shots, len(cls_ds)))))
        eval_sets.append(cls_ds.select(range(num_shots, min(num_shots*2 + 50, len(cls_ds)))))
    
    train_ds = concatenate_datasets(train_sets).shuffle(seed=seed)
    eval_ds = concatenate_datasets(eval_sets).shuffle(seed=seed)
    
    # Tokenize 映射函数
    def tokenize_fn(ex):
        return tokenizer(ex[text_col], padding=False, truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True).rename_column(label_col, "labels")
    eval_ds = eval_ds.map(tokenize_fn, batched=True).rename_column(label_col, "labels")
    
    cols = ['input_ids', 'attention_mask', 'labels']
    train_ds.set_format('torch', columns=cols)
    eval_ds.set_format('torch', columns=cols)
    
    return train_ds, eval_ds, len(unique_labels)
