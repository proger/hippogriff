import argparse
from contextlib import nullcontext
from pathlib import Path
from string import Template
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from hippogriff import GriffinLM, GriffinConfig
from train_diagnostics import summarize_activations, print_weights, summarize_gradients
from train_tape import Tapes
from train_checkpoints import list_checkpoints, load_checkpoint, save_checkpoint


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter): pass
parser = argparse.ArgumentParser("train", formatter_class=Formatter)
parser.add_argument('--exp', type=Template, default='exp', help="path to experiment directory (with substitution of $lr)")
parser.add_argument('--data', type=str, default='enwik8', help="collection of datasets to use for training, see classmethods in train_tapes.py for selection")
parser.add_argument('--lr', type=float, default=25e-5, help="learning rate")
parser.add_argument('--until', type=int, required=False, help="truncate run after this many steps")
parser.add_argument('--max_checkpoints', type=int, default=10, help="keep only the last n checkpoints")
parser.add_argument('--resume', action='store_true', help="resume from checkpoint")
parser.add_argument('--accumulate', type=int, default=1, help="accumulate gradients over this many steps")
parser.add_argument('--cooldown', type=str, choices=['linear', 'cosine'], default='linear', help="learning rate cooldown schedule")
parser.add_argument('--warmup', type=int, default=0, help="warmup steps")
parser.add_argument('--steps', type=int, default=100000, help="number of training steps")
parser.add_argument('--init', type=Path, help="load model weights from this checkpoint")
parser.add_argument('--seed', type=int, default=-1, help="random seed for the train data tape, defaults to sequential sampling when negative")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--eval_interval', type=int, default=1000, help="evaluate every n steps")

device = 'cuda' # use CUDA_VISIBLE_DEVICES to choose the device until accelerated-scan supports cuda:1


def make_model(vocab_size, *, args, seq_len=512):
    torch.manual_seed(1337)

    config = GriffinConfig(vocab_size=vocab_size)
    model = GriffinLM(config).to(device)

    if args.init:
        load_checkpoint(args.init, model=model, strict=False)

    print(model)
    print(f'entropy {np.log(config.vocab_size):.3f}')
    return model


@torch.inference_mode()
def evaluate(model, batches) -> tuple[float, dict]:
    model.eval()
    losses = []
    diag = {}
    for i, (input_ids, targets) in enumerate(batches):
        with summarize_activations(model, infix=['input', 'output'], verbose=i==0) as batch_diag:
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
            #with nullcontext():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()

            if i == 0:
                diag.update(batch_diag)
        losses.append(loss.item())
        if i and i % 100 == 0:
            print('mean bpc so far', np.mean(losses) / np.log(2))
    return np.mean(losses), diag


def train(model, tapes, opt, *, args):
    #torch.autograd.set_detect_anomaly(True)
    model.train()
    opt.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    step_tokens, total_tokens = 0, 0

    step = 0
    if args.resume:
        if checkpoints := list_checkpoints(args):
            step, total_tokens = load_checkpoint(checkpoints[-1], model=model, opt=opt, scaler=scaler, generator=tapes.train.generator)
    print_weights(model, full=True)

    steps = args.steps
    warmup_steps = args.warmup

    input_ids, targets = tapes.train[step]
    now = time.monotonic()
    for step in range(step, steps):
        step = step + 1
        for accumulation_step in range(0, args.accumulate):
            step_tokens += targets.numel()
            input_ids, targets = tapes.train[step]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()

        scaler.scale(loss/args.accumulate).backward()

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.25)

        if step < warmup_steps:
            # linear warmup
            current_lr = (step/warmup_steps) * args.lr
        elif args.cooldown == 'linear':
            lr_scale = (1 - (step-warmup_steps)/(steps-warmup_steps))
            current_lr = lr_scale * args.lr
        elif args.cooldown == 'cosine':
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * (step-warmup_steps)/(steps-warmup_steps)))
            current_lr = lr_scale * args.lr
        else:
            current_lr = args.lr
        opt.param_groups[0]['lr'] = current_lr

        scaler.step(opt)
        scaler.update()
        if step % args.eval_interval == 0:
            diag.update(summarize_gradients(model))
            diag.update(print_weights(model, full=True))
            # summarize optimizer updates?

        opt.zero_grad(set_to_none=True)

        diag = {}
        if step < 100 or step % 100 == 0:
            then = time.monotonic()
            total_tokens += step_tokens
            print(f'{step:6} steps, {total_tokens:9} tokens, {loss:.4f} xent, {loss/np.log(2):.4f} bpc,',
                    f'{current_lr:.5f} lr,',
                    f'{grad_norm:.4f} grad norm, {then-now:.4f} elapsed, {step_tokens/(then-now):.2f} tok/s', flush=True)
            diag.update({
                'train/loss': loss.item(),
                'train/bpc': loss.item() / np.log(2),
                'train/lr': current_lr,
                'train/grad_norm': grad_norm,
                'train/tps': step_tokens/(then-now),
                'train/total_tokens': total_tokens,
            })
            now = then
            step_tokens = 0

        if step % args.eval_interval == 0:
            save_checkpoint(model, opt, scaler, tapes.train.generator, step, total_tokens, args)
            eval_loss, activations = evaluate(model, tapes.valid)
            diag.update(activations)
            print(f'evaluate {eval_loss:.3f}', f'bpc {eval_loss / np.log(2):.3f}', 'after', step, 'steps', flush=True)
            diag.update({
                'eval/loss': eval_loss,
                'eval/bpc': eval_loss / np.log(2),
            })
            test_loss, _ = evaluate(model, tapes.test)
            print(f'test {test_loss:.3f}', f'bpc {test_loss / np.log(2):.3f}', 'after', step, 'steps', flush=True)
            diag.update({
                'test/loss': test_loss,
                'test/bpc': test_loss / np.log(2),
            })
            model.train()

        if diag and wandb.run is not None:
            wandb.log(diag, step=step)

        if args.until is not None and step >= args.until:
            print('stopping')
            model.eval()
            return

    model.eval()


if __name__ == '__main__':
    from tqdm import tqdm
    args = parser.parse_args()

    args.exp = Path(args.exp.substitute(**vars(args)))
    args.exp.mkdir(parents=True, exist_ok=True)
    with open(args.exp / 'run', 'a') as f:
        print(*sys.argv, file=f)

    tapes = getattr(Tapes, args.data)(args)
    model = make_model(tapes.vocab_size, seq_len=tapes.seq_len, args=args)
    opt = torch.optim.AdamW(model.parameter_groups(), lr=args.lr, betas=(0.9, 0.999), fused=False)

    args.parameters = sum(p.numel() for p in model.parameters())
    wandb.init(project='hippogriff', config=vars(args))

    train(model, tapes, opt, args=args)

    step = 'final'
    print('testing', flush=True)
    test_loss, _ = evaluate(model, tapes.test)
    print(f'final test {test_loss:.3f}', f'bpc {test_loss / np.log(2):.3f}', 'after', step, 'steps', flush=True)
    eval_loss, _ = evaluate(model, tqdm(tapes.valid))
    print(f'final evaluate {eval_loss:.3f}', f'bpc {eval_loss / np.log(2):.3f}', 'after', step, 'steps', flush=True)

    if wandb.run is not None:
        wandb.log({
            'final/eval/loss': eval_loss,
            'final/eval/bpc': eval_loss / np.log(2),
            'final/test/loss': test_loss,
            'final/test/bpc': test_loss / np.log(2),
        })
