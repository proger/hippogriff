"""
How much does the constant -8 matter in Hawk?
"""
from pathlib import Path
import torch
import wandb
import math

from train import train, Tapes, parser, device
from hippogriff import GriffinLM, GriffinConfig


def make_model(alpha_log_scale, vocab_size=16384, device='cuda'):
    torch.manual_seed(1337)

    config = GriffinConfig(vocab_size=vocab_size, smqa_head_dim=0)
    model = GriffinLM(config).to(device)

    print('initializing alpha_log_scale to', alpha_log_scale)
    for name, param in model.named_parameters():
        if 'alpha_log_scale' in name:
            if alpha_log_scale == 'learn':
                param.requires_grad = True
            else:
                param.data.fill_(math.log(alpha_log_scale))
                param.requires_grad = False
    
    return model


def run():
    args = parser.parse_args()
    args.exp = Path(args.exp.substitute(**vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project='hippogriff', config=vars(args))
    tapes = Tapes.languini(batch_size=args.batch_size)

    model = make_model(alpha_log_scale=wandb.config.alpha_log_scale, vocab_size=tapes.vocab_size, device=device)
    wandb.config.parameters = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)
    train(model, tapes, opt, args=args)


sweep_configuration = {
    "name": "alpha_log_scale",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "eval/loss"},
    "parameters": {
        "alpha_log_scale": {"values": ["learn", 14, 8, 4]},
    },
}


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="hippogriff")
    wandb.agent(sweep_id, function=run)


