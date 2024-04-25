"""
How do different models perform on a task of recalling the whole sequence with positional encodings and without?
"""
from pathlib import Path
import os
import torch
import wandb

from train import train, parser, device
from train_tape import Tapes
from hippogriff import GriffinLM, GriffinConfig
from multiquery_ar import sequence_recall

WANDB_PROJECT = 'hippogriff-mem-stacked'


def make_sequence_recall_tapes(num_examples=100_000):
    permuted = wandb.config.permuted
    vocab_size = wandb.config.vocab_size
    batch_size = wandb.config.batch_size
    seq_len = wandb.config.seq_len * 2 # double the sequence length due to stacking
    num_train_batches = num_examples // batch_size
    num_train_examples = num_train_batches*batch_size
    num_valid_batches = 3_000 // batch_size
    num_valid_examples = num_valid_batches*batch_size
    valid_inputs, valid_targets, _ = sequence_recall(vocab_size=vocab_size, num_examples=num_valid_examples, input_seq_len=seq_len, seed=43, stacked=True, permuted=permuted)
    train_inputs, train_targets, vocab_size = sequence_recall(vocab_size=vocab_size, num_examples=num_train_examples, input_seq_len=seq_len, seed=42, stacked=True, permuted=permuted)

    class Repeat:
        def __init__(self, inputs, targets, count=100000):
            self.inputs = inputs
            self.targets = targets
            self.count = count

        def __len__(self):
            return len(self.inputs) * self.count

        def __getitem__(self, i):
            input, target = self.inputs[i % len(self.inputs)], self.targets[i % len(self.targets)]
            return input.long(), target.long()

    tapes = Tapes(
        vocab_size=vocab_size,
        seq_len=seq_len,
        train=Repeat(train_inputs.view(num_train_batches, batch_size, -1, 2).to(device),
                     train_targets.view(num_train_batches, batch_size, -1).to(device)),
        valid=Repeat(valid_inputs.view(num_valid_batches, batch_size, -1, 2).to(device),
                     valid_targets.view(num_valid_batches, batch_size, -1).to(device), count=1),
    )
    print('mem: one epoch takes', num_train_batches, 'steps')

    i, t = tapes.train[0]
    print(i.shape, t.shape, 'train shapes')
    print('effective vocab size', vocab_size)
    return tapes, vocab_size


def run():
    wandb.init(project=WANDB_PROJECT)

    # how to nicely merge args and wandb.config?
    args = parser.parse_args()
    args.exp = Path(args.exp.substitute(run_id=wandb.run.id, **vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    args.lr = wandb.config.lr
    args.steps = wandb.config.steps

    tapes, vocab_size = make_sequence_recall_tapes(wandb.config.num_examples)

    torch.manual_seed(wandb.config.seed)

    dim = wandb.config.dim
    match wandb.config.model.split('_'):
        case ['hawk', 'noconv']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        # all s6 variants have conv turned off for now
        case ['s6', 'dstate1']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='S6', state_expansion=1,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate2']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=2, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate4']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=4, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate8']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=8, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate16']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=16, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate32']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=32, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['s6', 'dstate64']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0,
                                   time_module='S6', state_expansion=64, dim=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['qlstm']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=dim,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['qlstm', 'tied8']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=8,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['qlstm', 'tied16']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=16,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['qlstm', 'tied32']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='TiedQuasiLSTM', tied_quasi_lstm_num_heads=32,
                                   conv_kernel_size=0, hawk_expansion_factor=1)
        case ['outer', n, 'value']:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='OuterProduct', tied_quasi_lstm_num_heads=int(n),
                                   conv_kernel_size=0, hawk_expansion_factor=1, outer_query_values=True)
        case ['outer', n]:
            config = GriffinConfig(vocab_size=vocab_size, num_layers=wandb.config.num_layers, smqa_head_dim=0, dim=dim,
                                   time_module='OuterProduct', tied_quasi_lstm_num_heads=int(n),
                                   conv_kernel_size=0, hawk_expansion_factor=1)
    model = GriffinLM(config).to(device)
    print(model)
    wandb.config.parameters = sum(p.numel() for p in model.parameters())
    if config.time_module == 'OuterProduct':
        state_size = (config.dim // config.tied_quasi_lstm_num_heads)**2 * config.tied_quasi_lstm_num_heads
    else:
        state_size = config.dim
    wandb.config.state_size = sum([config.hawk_expansion_factor * config.state_expansion * state_size for _ in range(config.num_layers)])
    wandb.watch(model, log='all')

    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)
    train(model, tapes, opt, args=args)


sweep_configuration = {
    "name": "mem:len=64-512:vocab=1024,2048:dim=64:seeds3:100k",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "eval/accuracy"},
    "parameters": {
        "model": {"values": [
            #"s6_dstate8",
            #"s6_dstate16",
            #"s6_dstate32",
            "outer_8",
            "outer_4",
            "outer_2",
        ]},
        "dim": {"values": [64]},
        "num_layers": {"values": [1]},
        "num_examples": {"values": [100_000]},
        "lr": {"values": [2e-3]},
        "vocab_size": {"values":[1024]},
        "seq_len": {"values":[64,128,256,512]},
        "batch_size": {"values":[64]},
        "seed": {"values": [1,2,3]},
        "steps": {"values": [100_000]},
    },
}


if __name__ == '__main__':
    sweep_id = os.environ.get('SWEEP_ID')
    if not sweep_id:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJECT)
        print()
        print(f'Now start the agent with env SWEEP_ID={sweep_id}')
    else:
        wandb.agent(sweep_id, function=run)

