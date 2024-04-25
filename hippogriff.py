__version__ = '0.0.3'

from dataclasses import dataclass
import math
from typing import Literal
import torch
import torch.nn as nn
from torch.nn.functional import softplus, gelu, silu
from accelerated_scan.warp import scan
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


@dataclass
class GriffinConfig:
    vocab_size: int = 256
    num_layers: int = 1
    dim: int = 512
    smqa_head_dim: int = 128 # 0 for no attention
    smqa_q_heads: int = 4
    smqa_kv_heads: int = 1
    smqa_window_size: int = 512
    hawk_expansion_factor: float = 1.5 # parameteric state expansion, supported by: 'Hawk', 'S6'
    conv_kernel_size: int = 4 # supported by: 'Hawk', 'S6'
    time_module: Literal['TiedQuasiLSTM', 'AFWP', 'OuterProduct', 'Hawk', 'S6'] = 'Hawk'
    tied_quasi_lstm_num_heads: int = 16 # parameter-shared state expansion, supported by: 'TiedQuasiLSTM', 'AFWP', 'OuterProduct'
    state_expansion: int = 1 # parameter-shared state expansion, supported by: 'S6'
    gmlp_expansion_factor: float = 2
    outer_query_values: bool = False # perform final query from along the value dimension, supported by 'AFWP', 'OuterProduct'


class RMSNorm(nn.Module):
    def __init__(self, *, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return self.gamma / self.scale * x


class S6(nn.Module):
    def __init__(self, dim, expansion_factor=2, conv_kernel_size=4, d_state=16, n_layers=1, conv_bias=True, slow=False):
        super().__init__()
        self.hidden = hidden = int(dim * expansion_factor)
        self.delta_rank = math.ceil(dim / 16)
        self.d_state = d_state # state expansion factor
        self.slow = slow

        self.in_proj = nn.Linear(dim, 2*hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, self.delta_rank + d_state*2, bias=False)

        if conv_kernel_size:
            self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=bool(conv_bias),
                                  kernel_size=conv_kernel_size, groups=hidden, padding=conv_kernel_size - 1)
        else:
            self.conv = None

        self.delta_proj = nn.Linear(self.delta_rank, hidden, bias=False)

        dt_max = math.log(0.1)
        dt_min = math.log(0.001)
        dt_floor = 0.0001
        dt = torch.exp(torch.rand(hidden) * (dt_max - dt_min) + dt_min).clamp(min=dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.delta_bias = nn.Parameter(inv_dt, requires_grad=True)

        self.A_log = nn.Parameter(torch.arange(1, d_state+1, dtype=torch.float32).log().repeat(hidden, 1))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(hidden, dtype=torch.float32), requires_grad=True)
        self.out_proj = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            nn.init.uniform_(self.in_proj.weight, -dim**-0.5, dim**-0.5)
            nn.init.uniform_(self.gate_proj.weight, -hidden**-0.5, hidden**-0.5)
            nn.init.uniform_(self.delta_proj.weight, -self.delta_rank**-0.5, self.delta_rank**-0.5)
            nn.init.kaiming_uniform_(self.out_proj.weight, a=5**0.5)
            self.out_proj.weight /= n_layers**0.5

    def forward(self, x):
        u, o = self.in_proj(x).split(self.hidden, dim=-1)
        N, T, C = u.shape

        if self.conv is not None:
            u = self.conv(u.mT)[..., :T].mT
        u = silu(u)

        delta_lo, b, c = self.gate_proj(u).split([self.delta_rank, self.d_state, self.d_state], dim=-1)

        if self.slow:
            delta = (self.delta_proj(delta_lo) + self.delta_bias[None, None, :].float()).exp().log1p()

            delta = delta[..., None]
            forget = (-self.A_log.exp() * delta).exp()
            update = delta * u.unsqueeze(-1) * b.unsqueeze(-2)
            forget = forget.view(N, T, C*self.d_state).mT.contiguous()
            update = update.view(N, T, C*self.d_state).mT.contiguous()
            h = scan(forget, update)
            h = h.mT.contiguous().view(N, T, C, self.d_state)
            y = (c.unsqueeze(-2) * h).sum(dim=-1)

            y = y + self.D * u
            y = y * o.sigmoid() * o
        else:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            delta = self.delta_proj(delta_lo.float()).mT
            a = -self.A_log.exp().float()
            y = selective_scan_fn(u.mT, delta, A=a, B=b.mT, C=c.mT, D=self.D, z=o.mT,
                                  delta_bias=self.delta_bias.float(), delta_softplus=True)
            y = y.mT

        return self.out_proj(y)


class TiedQuasiLSTM(nn.Module):
    def __init__(self, *, dim, num_heads):
        super().__init__()
        self.head_dim = dim // num_heads
        self.hidden = hidden = self.head_dim * num_heads
        self.num_heads = num_heads
        self.gates = nn.Linear(dim, num_heads + 3 * hidden, bias=True) # bias for the gates
        self.output = nn.Linear(hidden, dim)

        with torch.no_grad():
            self.gates.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        f, i, t, o = self.gates(x).split([self.num_heads, self.hidden, self.hidden, self.hidden], dim=-1)
        f = f.sigmoid().repeat_interleave(self.head_dim, -1)
        update = i.sigmoid() * t.tanh()
        c = scan(f.mT.contiguous(), update.mT.contiguous()).mT
        h = o * torch.tanh(c)
        x = self.output(h)
        return x


class AFWP(nn.Module):
    def __init__(self, *, key_dim, value_dim, num_heads, outer_query_values=False):
        super().__init__()
        model_dim = value_dim * num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.outer_query_values = outer_query_values
        if outer_query_values:
            self.query_dim = value_dim
            self.hidden_dim = key_dim
        else:
            self.query_dim = key_dim
            self.hidden_dim = value_dim
        self.key = nn.Linear(model_dim, self.key_dim * num_heads, bias=True) # bias for the gate
        self.query = nn.Linear(model_dim, self.query_dim * num_heads, bias=False)
        self.output = nn.Linear(self.hidden_dim * num_heads, model_dim, bias=False)

        with torch.no_grad():
            self.key.weight.normal_(std=model_dim**-0.5)
            self.key.bias.uniform_(-4, 4)
            self.query.weight.normal_(std=model_dim**-0.5)
            self.output.weight.normal_(std=(self.hidden_dim * num_heads)**-0.5)

    def forward(self, x):
        N, T, HV = x.shape
        q = self.query(x)
        k = self.key(k).sigmoid()
        k = k.view(N, T, self.num_heads, -1) # N, T, H, K
        x = x.view(N, T, self.num_heads, -1) # N, T, H, V
        kv = k.unsqueeze(-1) * x.unsqueeze(-2) # outer product, N, T, H, K, V
        kv = kv.view(N, T, -1) # N, T, H*K*V
        k = (1-k).repeat_interleave(self.value_dim, dim=-1) # N, T, H, K*V
        k = k.view(N, T, -1) # N, T, H*K*V
        w = scan(k.mT.contiguous(), kv.mT.contiguous()).mT # fast weights
        w = w.reshape(N, T, self.num_heads, self.key_dim, -1) # N, T, H, K, V
        if self.outer_query_values:
            q = q.view(N, T, self.num_heads, self.query_dim, 1) # N, T, H, V, 1
            h = w @ q # N, T, H, K, 1
        else:
            q = q.view(N, T, self.num_heads, 1, self.query_dim) # N, T, H, 1, K
            h = q @ w # N, T, H, 1, V
        return self.output(h.view(N, T, -1))


class OuterProduct(nn.Module):
    def __init__(self, *, dim, num_heads, outer_query_values=False):
        super().__init__()
        self.head_dim = dim // num_heads
        self.hidden = hidden = self.head_dim * num_heads
        self.num_heads = num_heads
        self.key = nn.Linear(dim, hidden, bias=True) # bias for the gate
        self.vq = nn.Linear(dim, 2 * hidden, bias=False)

        self.output = nn.Linear(hidden, dim)
        self.outer_query_values = outer_query_values

        with torch.no_grad():
            self.key.weight.normal_(std=dim**-0.5)
            self.key.bias.uniform_(-4, 4)
            self.vq.weight.normal_(std=dim**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        N, T, HD = x.shape
        v, q = self.vq(x).chunk(2, dim=-1)
        k = self.key(x).sigmoid()
        k = k.view(N, T, self.num_heads, self.head_dim) # N, T, H, K
        v = v.view(N, T, self.num_heads, self.head_dim) # N, T, H, V
        kv_update = (1 - k).unsqueeze(-1) * v.unsqueeze(-2) # N, T, H, K, V
        kv_update = kv_update.view(N, T, -1) # N, T, H*K*V
        k = k.repeat_interleave(self.head_dim, dim=-1) # N, T, H, K*V
        k = k.view(N, T, -1) # N, T, H*K*V
        kv = scan(k.mT.contiguous(), kv_update.mT.contiguous()).mT
        kv = kv.reshape(N, T, self.num_heads, self.head_dim, self.head_dim) # N, T, H, K, V
        if self.outer_query_values:
            q = q.view(N, T, self.num_heads, self.head_dim, 1) # N, T, H, V, 1
            h = kv @ q
        else:
            q = q.view(N, T, self.num_heads, 1, self.head_dim) # N, T, H, 1, K
            h = q @ kv
        return self.output(h.view(N, T, HD))



class Hawk(nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=1.5, conv_kernel_size=4):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = nn.Linear(dim, 2*hidden, bias=False)
        if conv_kernel_size:
            self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                                  kernel_size=conv_kernel_size, groups=hidden, padding=conv_kernel_size-1)
        else:
            self.conv = None
        self.gates = nn.Linear(hidden, 2*hidden, bias=True)
        def mk(hidden, a=0.001, b=0.1, lo=-4.323, hi=-9):
            x = torch.log(torch.expm1(torch.linspace(a, b, hidden)))
            x = (x - x.min()) / (x.max() - x.min())
            x = x * abs(hi-lo) + hi
            return x
        self.forget_base = nn.Parameter(mk(hidden))
        self.output = nn.Linear(hidden, dim, bias=False)
        self.alpha_log_scale = nn.Parameter(torch.tensor([8]).log(), requires_grad=False)

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.gates.weight.normal_(std=hidden**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        _N, T, _C = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        if self.conv is not None:
            x = self.conv(x.mT)[..., :T].mT

        # RG-LRU: linear recurrent unit with input-dependent gating
        forget, input = self.gates(x).chunk(2, dim=-1)
        alpha = (-self.alpha_log_scale.exp() * softplus(self.forget_base) * forget.sigmoid()).exp()
        beta = (1 - alpha**2 + 1e-6).sqrt() # stabilizes variance
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


class Block(nn.Module):
    def __init__(self, config: GriffinConfig):
        super().__init__()

        self.attention = config.smqa_head_dim > 0
        if self.attention:
            self.smqa_norm = RMSNorm(dim=config.dim)
            self.smqa = SlidingMQA(dim=config.dim, head_dim=config.smqa_head_dim, q_heads=config.smqa_q_heads,
                                kv_heads=config.smqa_kv_heads, window_size=config.smqa_window_size)
            self.smqa_gmlp_norm = RMSNorm(dim=config.dim)
            self.smqa_gmlp = GatedMLP(dim=config.dim, expansion_factor=config.gmlp_expansion_factor)

        self.time_norm = RMSNorm(dim=config.dim)
        match config.time_module:
            case 'TiedQuasiLSTM':
                self.time = TiedQuasiLSTM(dim=config.dim, num_heads=config.tied_quasi_lstm_num_heads)
            case 'AFWP':
                self.time = AFWP(key_dim=config.dim, value_dim=config.dim, num_heads=config.tied_quasi_lstm_num_heads, outer_query_values=config.outer_query_values)
            case 'OuterProduct':
                self.time = OuterProduct(dim=config.dim, num_heads=config.tied_quasi_lstm_num_heads, outer_query_values=config.outer_query_values)
            case 'Hawk':
                self.time = Hawk(dim=config.dim, expansion_factor=config.hawk_expansion_factor, conv_kernel_size=config.conv_kernel_size)
            case 'S6':
                self.time = S6(dim=config.dim, expansion_factor=config.hawk_expansion_factor, conv_kernel_size=config.conv_kernel_size,
                               d_state=config.state_expansion, n_layers=config.num_layers, conv_bias=True, slow=False)
        self.gmlp_norm = RMSNorm(dim=config.dim)
        self.gmlp = GatedMLP(dim=config.dim, expansion_factor=config.gmlp_expansion_factor)

    def forward(self, x):
        if self.attention:
            x = x + self.smqa(self.smqa_norm(x))
            x = x + self.smqa_gmlp(self.smqa_gmlp_norm(x))
        x = x + self.time(self.time_norm(x))
        x = x + self.gmlp(self.gmlp_norm(x))
        return x


class GriffinLM(nn.Module):
    def __init__(self, config: GriffinConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.backbone = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.output_norm = RMSNorm(dim=config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        with torch.no_grad():
            self.embedding.weight.normal_(std=config.dim**-0.5)
            self.lm_head.weight.normal_(std=config.dim**-0.5)

        self.tie_weights_()

    def tie_weights_(self):
        self.lm_head.weight = self.embedding.weight

    def parameter_groups(self, weight_decay=1e-2):
        return [
            {'params': self.embedding.parameters(), 'weight_decay': 0.0}, # lm_head is tied here
            # do not decay biases and single-column parameters (forget_base, rmsnorm), those are usually scales
            {'params': (p for p in self.backbone.parameters() if p.dim() < 2 or getattr(p, '_no_weight_decay', False)), 'weight_decay': 0.0},
            {'params': (p for p in self.backbone.parameters() if p.dim() >= 2 and not getattr(p, '_no_weight_decay', False)), 'weight_decay': weight_decay},
            {'params': self.output_norm.parameters(), 'weight_decay': 0.0},
        ]

    def forward(self, input_ids):
        N, T, *rest = input_ids.shape
        x = self.embedding(input_ids)
        if rest:
            x = x.sum(dim=tuple(range(2, 2+len(rest)))) # marginalize extra dimensions if present
        for block in self.backbone:
            x = block(x)
        x = self.lm_head(self.output_norm(x))
        return x


if __name__ == '__main__':
    device = 'cuda'
    torch.manual_seed(3407)

    config = GriffinConfig(time_module='S6', smqa_head_dim=0)
    model = GriffinLM(config).to('cuda')
    print(model)
    input_ids = torch.randint(0, config.vocab_size, (1, 1024), device='cuda')
    with torch.amp.autocast(device_type='cuda'):
        from train_diagnostics import summarize_activations
        with summarize_activations(model, infix=['proj', 'conv'], verbose=True) as log:
            output = model(input_ids)
        for k in log:
            if 'mean/' in k:
                print(k, log[k])
        probs = output.softmax(dim=-1)
    print(probs.shape)
