"""
How do different models perform on the multi-query associative recall task?
"""
from pathlib import Path
import torch
import wandb

from train import train, parser, device
from train_tape import Tapes
from hippogriff import GriffinLM, GriffinConfig
from multiquery_ar import multiquery_ar

WANDB_PROJECT = 'hippogriff-mqar'

def run():
    wandb.init(project=WANDB_PROJECT)

    # how to nicely merge args and wandb.config?
    args = parser.parse_args()
    args.exp = Path(args.exp.substitute(**vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    args.lr = wandb.config.lr

    vocab_size = 64
    batch_size = 64
    num_train_batches = 100_000 // batch_size
    num_valid_batches = 3_000 // batch_size
    seq_len = 64
    train_inputs, train_targets = multiquery_ar(vocab_size=vocab_size, num_examples=num_train_batches*batch_size, input_seq_len=seq_len, seed=42, power_a=0.01, num_kv_pairs=8, random_non_queries=False)
    valid_inputs, valid_targets = multiquery_ar(vocab_size=vocab_size, num_examples=num_valid_batches*batch_size, input_seq_len=seq_len, seed=43, power_a=0.01, num_kv_pairs=8, random_non_queries=False)

    class Repeat:
        def __init__(self, xs):
            self.xs = xs

        def __getitem__(self, i):
            return self.xs[i % len(self.xs)]
    
    tapes = Tapes(
        vocab_size=vocab_size,
        seq_len=seq_len,
        train=Repeat([(input, target) for input, target in zip(train_inputs.to(device).view(num_train_batches, batch_size, seq_len),
                                                               train_targets.to(device).view(num_train_batches, batch_size, seq_len))]),
        valid=[(input, target) for input, target in zip(valid_inputs.to(device).view(num_valid_batches, batch_size, seq_len),
                                                        valid_targets.to(device).view(num_valid_batches, batch_size, seq_len))],
    )
    print('mqar: one epoch takes', len(tapes.train.xs), 'steps')

    torch.manual_seed(1337)

    dim = wandb.config.dim
    match wandb.config.model.split('_'):
        case ['hawk']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim)
        case ['qlstm']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=dim)
        case ['qlstm', 'tied8']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=8)
        case ['qlstm', 'tied16']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=16)
        case ['qlstm', 'tied32']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=32)
    model = GriffinLM(config).to(device)
    wandb.config.parameters = sum(p.numel() for p in model.parameters())
    wandb.watch(model, log='all')

    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)
    train(model, tapes, opt, args=args)


sweep_configuration = {
    "name": "mqar+lr",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "eval/accuracy"},
    "parameters": {
        "model": {"values": ["hawk", "qlstm", "qlstm_tied8", "qlstm_tied16"]},
        "dim": {"values": [256]},
        "num_layers": {"values": [2]},
        "lr": {"values": [1e-2, 1e-3, 3e-4, 1e-4]},
    },
}

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=run)


