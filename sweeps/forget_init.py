"""
How much does the shape of the forget_base initialization matter in Hawk?
"""
from pathlib import Path
import torch
import wandb
import math

from train import train, Tapes, parser, device
from hippogriff import GriffinLM, GriffinConfig


def make_model(forget_init, vocab_size=16384, device='cuda'):
    torch.manual_seed(1337)

    config = GriffinConfig(vocab_size=vocab_size, smqa_head_dim=0)
    model = GriffinLM(config)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'forget_base' in name:
                if forget_init == 'uniform':
                    def mk(a=0.001, b=0.1, lo=-4.323, hi=-9):
                        x = torch.log(torch.expm1(torch.linspace(a, b, param.numel())))
                        x = (x - x.min()) / (x.max() - x.min())
                        x = x * abs(hi-lo) + hi
                        return x

                    # initialize forget_base so
                    # alpha = (-alpha_log_scale.exp() * softplus(forget_base)).exp()
                    # looks uniform (similar to the griffin paper)
                    # then sigmoid makes the whole alpha look like steps

                    param.copy_(mk())
                elif forget_init == 'exp':
                    param.copy_(torch.linspace(-4.323, -9, param.numel()))

    model = model.to(device)
    return model


def run():
    args = parser.parse_args()
    args.exp = Path(args.exp.substitute(**vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project='hippogriff', config=vars(args))
    tapes = Tapes.languini(batch_size=args.batch_size)

    model = make_model(forget_init=wandb.config.forget_init, vocab_size=tapes.vocab_size, device=device)
    wandb.config.parameters = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)
    train(model, tapes, opt, args=args)


sweep_configuration = {
    "name": "forget_init",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "eval/loss"},
    "parameters": {
        "forget_init": {"values": ["uniform", "exp"]},
    },
}


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="hippogriff")
    wandb.agent(sweep_id, function=run)


