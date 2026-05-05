import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer

# ===================== 评估函数 =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average="macro")}

# ===================== 自定义 SCL Trainer =====================
class DualPromptSCLTrainer(Trainer):
    def __init__(self, scl_alpha=0.2, temperature=0.1, cls_index=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scl_alpha = scl_alpha
        self.temperature = temperature
        self.cls_index = cls_index

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # 前向传播，获取 hidden_states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        ce_loss = outputs.loss
        
        # 验证阶段或 alpha 为 0 时，仅计算交叉熵
        if not model.training or self.scl_alpha == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # 获取最后一层的隐状态
        hidden_states = outputs.hidden_states[-1]
        
        # 提取 CLS representation (由于前置了 prompt token，需要偏移 cls_index)
        features = hidden_states[:, self.cls_index, :]
        features = F.normalize(features, p=2, dim=-1)

        # 提取全局可学习的原型 (Prototypes) 并计算余弦相似度
        proto_norm = F.normalize(model.global_prototypes, p=2, dim=-1)
        logits_cl = torch.matmul(features, proto_norm.T) / self.temperature
        
        # 计算 Prototype-Centered SCL 损失
        scl_loss = F.cross_entropy(logits_cl, labels)
        
        # 联合优化目标
        loss = ce_loss + self.scl_alpha * scl_loss
        
        return (loss, outputs) if return_outputs else loss
