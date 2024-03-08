__version__ = '0.0.1'

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.functional import softplus, gelu
from accelerated_scan.warp import scan
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


@dataclass
class GriffinConfig:
    vocab_size = 256
    num_layers = 1
    dim = 1024
    smqa_head_dim = 128
    smqa_q_heads = 8
    smqa_kv_heads = 1
    smqa_window_size = 512
    hawk_expansion_factor = 1.5
    hawk_kernel_size = 4
    gmlp_expansion_factor = 2


class RMSNorm(nn.Module):
    def __init__(self, *, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        return self.gamma / self.scale * x


class Hawk(nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=1.5, kernel_size=4):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = nn.Linear(dim, 2*hidden, bias=False)
        self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = nn.Linear(hidden, 2*hidden, bias=True)
        self.forget_base = nn.Parameter(torch.linspace(-6.7, -10.5, hidden))
        self.output = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.gates.weight.normal_(std=hidden**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        _N, T, _C = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT

        # RG-LRU: linear recurrent unit with input-dependent gating
        forget, input = self.gates(x).chunk(2, dim=-1)
        alpha = (-8 * softplus(self.forget_base) * forget.sigmoid()).exp()
        beta = (1 - alpha**2).sqrt()
        x = beta * input.sigmoid() * x

        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        x = self.output(gelu(gate) * h)
        return x


class GatedMLP(nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = nn.Linear(dim, 2 * hidden, bias=False)
        self.shrink = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim**-0.5)
            self.shrink.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = gelu(gate) * x
        return self.shrink(x)


class SlidingMQA(nn.Module):
    def __init__(self, *, dim=1024, head_dim=128, q_heads=8, kv_heads=1, window_size=1024):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.rotary = RotaryEmbedding(dim=head_dim)
        self.q = nn.Linear(dim, head_dim*q_heads, bias=False)
        self.kv = nn.Linear(dim, 2*head_dim*kv_heads, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            self.q.weight.normal_(std=dim**-0.5)
            self.kv.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=dim**-0.5)

    def forward(self, x):
        N, T, C = x.shape
        q = self.q(x).view(N, T, -1, self.head_dim)
        kv = self.kv(x).view(N, T, 2, -1, self.head_dim)
        q, kv = self.rotary(q, kv)
        x = flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], causal=True, window_size=(-self.window_size, 0))
        x = x.view(N, T, C)
        return self.output(x)


class Griffin(nn.Module):
    def __init__(self, config: GriffinConfig):
        super().__init__()
        self.hawk_norm = RMSNorm(dim=config.dim)
        self.hawk = Hawk(dim=config.dim, expansion_factor=config.hawk_expansion_factor, kernel_size=config.hawk_kernel_size)
        self.hawk_gmlp_norm = RMSNorm(dim=config.dim)
        self.hawk_gmlp = GatedMLP(dim=config.dim, expansion_factor=config.gmlp_expansion_factor)

        self.smqa_norm = RMSNorm(dim=config.dim)
        self.smqa = SlidingMQA(dim=config.dim, head_dim=config.smqa_head_dim, q_heads=config.smqa_q_heads,
                               kv_heads=config.smqa_kv_heads, window_size=config.smqa_window_size)
        self.smqa_gmlp_norm = RMSNorm(dim=config.dim)
        self.smqa_gmlp = GatedMLP(dim=config.dim, expansion_factor=config.gmlp_expansion_factor)

    def forward(self, x):
        x += self.hawk(self.hawk_norm(x))
        x += self.hawk_gmlp(self.hawk_gmlp_norm(x))
        x += self.smqa(self.smqa_norm(x))
        x += self.smqa_gmlp(self.smqa_gmlp_norm(x))
        return x


class GriffinLM(nn.Module):
    def __init__(self, config: GriffinConfig):
        super().__init__()
        self.input = nn.Embedding(config.vocab_size, config.dim)
        self.griffin = nn.ModuleList([Griffin(config) for _ in range(config.num_layers)])
        self.output_norm = RMSNorm(dim=config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        with torch.no_grad():
            self.input.weight.normal_(std=config.dim**-0.5)
            self.output.weight.normal_(std=config.dim**-0.5)

        self.tie_weights_()

    def tie_weights_(self):
        self.output.weight = self.input.weight

    def parameter_groups(self):
        return [
            {'params': self.input.parameters()},
            {'params': self.griffin.parameters()},
            {'params': self.output.parameters()}
        ]

    def forward(self, input_ids):
        x = self.input(input_ids)
        for block in self.griffin:
            x = block(x)
        x = self.output(self.output_norm(x))
        return x

        


if __name__ == '__main__':
    device = 'cuda'
    torch.manual_seed(3407)

    config = GriffinConfig()
    griffin = GriffinLM(config).to('cuda')
    input_ids = torch.randint(0, config.vocab_size, (1, 1024), device='cuda')
    with torch.amp.autocast(device_type='cuda'):
        output = griffin(input_ids)
        probs = output.softmax(dim=-1)
    print(probs.shape)
