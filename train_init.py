import numpy as np
import torch
from hippogriff import GriffinLM, GriffinConfig


def make_model(vocab_size, *, init=None, device='cuda'):
    torch.manual_seed(1337)

    config = GriffinConfig(vocab_size=vocab_size)
    model = GriffinLM(config).to(device)

    if init:
        load_checkpoint(init, model=model, strict=False)

    print(model)
    print(f'entropy {np.log(config.vocab_size):.3f}')
    return model


def list_checkpoints(args):
    checkpoints = sorted(args.exp.glob('checkpoint.*.pt'), key=lambda c: c.stat().st_mtime)
    return checkpoints


def load_checkpoint(checkpoint, *, model, opt=None, scaler=None, generator=None, strict=True):
    ckpt = torch.load(checkpoint)
    print('resuming from', checkpoint, f'{strict=}')
    model.load_state_dict(ckpt['model'], strict=strict)
    if opt is not None:
        opt.load_state_dict(ckpt['optimizer'])
        print('loaded optimizer state')
    if scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])
        print('loaded scaler state')
    if generator is not None:
        generator.set_state(ckpt['tape'])
        print('loaded tape state')
    step = ckpt['step']+1
    total_tokens = ckpt['total_tokens']
    return step, total_tokens


def save_checkpoint(model, optimizer, scaler, generator, step, total_tokens, args):
    checkpoints = list_checkpoints(args)
    last_checkpoint = args.max_checkpoints-1
    if len(checkpoints) > last_checkpoint:
        for old_checkpoint in checkpoints[:-last_checkpoint]:
            old_checkpoint.unlink()
            print('removed', old_checkpoint)
    checkpoint_filename = args.exp / f'checkpoint.{step:05}.pt'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'generator': generator.get_state() if generator is not None else None,
        'step': step,
        'total_tokens': total_tokens,
        'args': vars(args),
    }, checkpoint_filename)
    print('saved', checkpoint_filename)
