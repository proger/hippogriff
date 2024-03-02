import torch
import torch.nn as nn
from torch.nn.functional import softplus, gelu
from accelerated_scan.warp import scan


class Hawk(nn.Module):
    def __init__(self, dim=1024, expansion_factor=1.5, kernel_size=4):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = nn.Linear(dim, 2*hidden, bias=False)
        self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = nn.Linear(hidden, 2*hidden, bias=True)
        self.forget_base = nn.Parameter(torch.linspace(-6.7, -10.5, hidden))
        self.output = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.input.weight.normal_(std=1/dim**0.5)
            self.gates.weight.normal_(std=1/hidden**0.5)
            self.output.weight.normal_(std=1/hidden**0.5)

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


if __name__ == '__main__':
    torch.manual_seed(3407)
    dim = 32
    hawk = Hawk(dim=dim).to('cuda')
    x = torch.randn(1, 2048, dim, device='cuda')
    print(hawk(x).shape)
