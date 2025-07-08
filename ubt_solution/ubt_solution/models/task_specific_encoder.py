import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..config import Config
# from training_pipeline.constants import HIDDEN_SIZE_THIN, HIDDEN_SIZE_WIDE

HIDDEN_SIZE_THIN=2048
HIDDEN_SIZE_WIDE=4096
# 本地复制 DLRM_v3 的 init_mlp_weights_optional_bias
def init_mlp_weights_optional_bias(m: torch.nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 本地实现 SwishLayerNorm
def swish_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    ln = F.layer_norm(
        x_fp32,
        [x_fp32.shape[-1]],
        weight.to(torch.float32),
        bias.to(torch.float32),
        eps,
    )
    return (x_fp32 * torch.sigmoid(ln)).to(dtype)

class SwishLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_layer_norm(x, self.weight, self.bias, self.eps)

# 添加与 training_pipeline 中相同的 BottleneckBlock 和 HeadNet
class BottleneckBlock(nn.Module):
    def __init__(self, thin_dim: int, wide_dim: int):
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

class HeadNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, thin_dim: int, wide_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, thin_dim)
        self.ln_input = nn.LayerNorm(thin_dim)
        self.layernorms = nn.ModuleList([nn.LayerNorm(thin_dim) for _ in range(3)])
        self.bottlenecks = nn.ModuleList([BottleneckBlock(thin_dim, wide_dim) for _ in range(3)])
        self.ln_output = nn.LayerNorm(thin_dim)
        self.linear_output = nn.Linear(thin_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_input(self.input_projection(x))
        for ln, bottleneck in zip(self.layernorms, self.bottlenecks):
            x = x + bottleneck(ln(x))
        x = self.ln_output(x)
        return self.linear_output(x)

class TaskSpecificEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config # Store config
        # 共享底层表示
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 任务头 - 与 training_pipeline 中 Net 结构一致（添加 price 回归头）
        self.task_heads = nn.ModuleDict({
            'churn': HeadNet(
                input_dim=config.hidden_size,
                output_dim=1,
                thin_dim=HIDDEN_SIZE_THIN,
                wide_dim=HIDDEN_SIZE_WIDE
            ),
            'price': HeadNet(
                input_dim=config.hidden_size,
                output_dim=1,
                thin_dim=HIDDEN_SIZE_THIN,
                wide_dim=HIDDEN_SIZE_WIDE
            ),
        })
        
        # 初始化权重（部分在 init_mlp_weights_optional_bias 中已处理）
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, module in self.task_heads.items():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, user_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        # 共享表示
        shared_features = self.shared_encoder(user_embedding)
        
        # 流失预测
        outputs['churn'] = self.task_heads['churn'](shared_features).squeeze(-1)

        # 价格回归预测
        outputs['price'] = self.task_heads['price'](shared_features).squeeze(-1)
        
        # 可选：对 churn 输出裁剪以稳定训练
        outputs['churn'] = torch.clamp(outputs['churn'], min=-10.0, max=10.0)
        
        return outputs 