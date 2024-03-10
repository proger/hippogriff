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

from train_diagnostics import summarize_activations, print_weights, summarize_gradients
from train_tape import Tapes
from train_init import list_checkpoints, load_checkpoint, save_checkpoint, make_model


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter): pass
parser = argparse.ArgumentParser("train", formatter_class=Formatter)
parser.add_argument('--exp', type=Template, default='exp', help="path to experiment directory (with substitution of other command line arguments using $lr syntax)")
parser.add_argument('--data', type=str, default='enwik8', help="collection of datasets to use for training, see classmethods in train_tapes.py for selection")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
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
parser.add_argument('--log_interval', type=int, default=100, help="log every n steps")
parser.add_argument('--eval_interval', type=int, default=1000, help="evaluate every n steps")
parser.add_argument('--anomaly', type=str, choices=['auto', 'active', 'ignore'], default='auto', help="when to detect and break on anomalies: auto (default) enables anomaly detection only when a nan gradient is detected, active enables anomaly detection for all steps, ignore disables anomaly detection.")

device = 'cuda' # use CUDA_VISIBLE_DEVICES to choose the device until accelerated-scan supports cuda:N
dtype = torch.bfloat16 # torch.float16


@torch.inference_mode()
def evaluate(model, batches, diag_prefix='eval') -> tuple[float, dict]:
    model.eval()
    losses = []
    accuracy_sum, accuracy_count = 0, 0
    diag = {}
    for i, (input_ids, targets) in enumerate(batches):
        with summarize_activations(model, infix=['input', 'output'], verbose=i==0) as batch_diag:
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
            #with nullcontext():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()

            accuracy_sum, accuracy_count = logits.argmax(dim=-1).eq(targets).sum().item(), targets.numel()

            if i == 0:
                diag.update(batch_diag)
        losses.append(loss.item())
        if i and i % 100 == 0:
            bpb = np.mean(losses) / np.log(2)
            print(f'evaluation step {i}: bpb so far {bpb:.4f}')
    diag.update({
        f'{diag_prefix}/accuracy': accuracy_sum / accuracy_count,
        f'{diag_prefix}/loss': np.mean(losses),
        f'{diag_prefix}/bpb': np.mean(losses) / np.log(2),
    })
    return diag


def train(model, tapes, opt, *, args):
    torch.autograd.set_detect_anomaly(args.anomaly == 'active')
    model.train()
    opt.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler(enabled=dtype==torch.float16)
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
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()

        scaler.scale(loss/args.accumulate).backward()

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.25)
        isnan_grad_norm = torch.isnan(grad_norm).any()
    
        if isnan_grad_norm and args.anomaly == 'auto':
            print('step', step, 'has a nan gradient, retrying with anomaly detector')
            summarize_gradients(model)
            opt.zero_grad(set_to_none=True)
            assert not len(summarize_gradients(model)), "some model parameters still have gradients after opt.zero_grad, check your optimizer parameter coverage"
            with torch.autograd.set_detect_anomaly(True):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()
                scaler.scale(loss).backward()

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
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr

        scaler.step(opt)
        scaler.update()

        diag = {}

        #if step % args.log_interval == 0 or isnan_grad_norm:
        if step % args.eval_interval == 0 or isnan_grad_norm:
            diag.update(summarize_gradients(model))
            diag.update(print_weights(model, full=True))
            # summarize optimizer updates?

        opt.zero_grad(set_to_none=True)

        if step < 100 or step % args.log_interval == 0:
            then = time.monotonic()
            total_tokens += step_tokens
            scaler_info = f'{scaler.get_scale()} scale, ' if scaler.is_enabled() else ''
            print(f'{step:6} steps, {total_tokens:9} tokens, {loss:.4f} xent, {scaler_info}{loss/np.log(2):.4f} bpb,',
                    f'{current_lr:.5f} lr,',
                    f'{grad_norm:.4f} grad norm, {then-now:.4f} elapsed, {step_tokens/(then-now):.2f} tok/s', flush=True)
            diag.update({
                'train/loss': loss.item(),
                'train/bpb': loss.item() / np.log(2),
                'train/lr': current_lr,
                'train/grad_norm': grad_norm,
                'train/tps': step_tokens/(then-now),
                'train/total_tokens': total_tokens,
                **{'train/scaler_' + k: v for k, v in scaler.state_dict().items()}
            })
            now = then
            step_tokens = 0

        if step % args.eval_interval == 0:
            save_checkpoint(model, opt, scaler, tapes.train.generator, step, total_tokens, args)
            eval = evaluate(model, tapes.valid, diag_prefix='eval')
            eval_loss, eval_bpb, eval_accuracy = eval['eval/loss'], eval['eval/bpb'], eval['eval/accuracy']
            diag.update(eval)
            print(f'evaluate xent {eval_loss:.3f}', f'bpb {eval_bpb:.3f}', f'accuracy {eval_accuracy:.3f}', 'after', step, 'steps', flush=True)
            if False:
                test = evaluate(model, tapes.test, diag_prefix='test')
                test_loss, test_bpb = test['test/loss'], test['test/bpb']
                print(f'test xent {test_loss:.3f}', f'bpb {test_bpb:.3f}', 'after', step, 'steps', flush=True)
                diag.update({
                    'test/loss': test_loss,
                    'test/bpb': test_bpb,
                    'test/accuracy': test['test/accuracy'],
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
    model = make_model(tapes.vocab_size, init=args.init, device=device)
    parameter_groups = model.parameter_groups()
    opt = torch.optim.AdamW(parameter_groups, lr=args.lr, betas=(0.9, 0.999), fused=False)
    for i, param_group in enumerate(opt.param_groups):
        n = sum(p.numel() for p in param_group['params'])
        print('parameter group', i, 'has', n, 'parameters')

    args.parameters = sum(p.numel() for p in model.parameters())
    wandb.init(project='hippogriff', config=vars(args))

    train(model, tapes, opt, args=args)

    if False:
        step = 'final'
        print('testing', flush=True)
        test = evaluate(model, tapes.test, diag_prefix='test')
        test_loss, test_bpb = test['test/loss'], test['test/bpb']
        print(f'final test xent {test_loss:.3f}', f'bpb {test_bpb:.3f}', 'after', step, 'steps', flush=True)
        eval = evaluate(model, tqdm(tapes.valid), diag_prefix='eval')
        eval_loss, eval_bpb = eval['eval/loss'], eval['eval/bpb']
        print(f'final evaluate xent {eval_loss:.3f}', f'bpb {eval_bpb:.3f}', 'after', step, 'steps', flush=True)

        if wandb.run is not None:
            wandb.log({
                'final/eval/loss': eval_loss,
                'final/eval/bpb': eval_bpb,
                'final/test/loss': test_loss,
                'final/test/bpb': test_bpb,
            })
