from contextlib import contextmanager
from functools import partial

import torch


@contextmanager
def summarize_activations(model, infix=[], verbose=False):
    log = {}
    i = 0
    def hook(module, args, output, *, name):
        nonlocal i
        x = args[0]
        y = output
        bsz, seq_len, i_dim = x.shape

        imean = x.mean().float().cpu().item()
        istd = x.std().float().cpu().item()
        omean = y.mean().float().cpu().item()
        ostd = y.std().float().cpu().item()
        inorml = x[:, :seq_len//2, :].norm(dim=2).mean().float().cpu().item()
        inormr = x[:, seq_len//2:, :].norm(dim=2).mean().float().cpu().item()
        onorml = y[:, :seq_len//2, :].norm(dim=2).mean().float().cpu().item()
        onormr = y[:, seq_len//2:, :].norm(dim=2).mean().float().cpu().item()

        if i < 16 and verbose:
            print(f"act name={name} shape={tuple(x.shape)} {imean=:>.4f} {istd=:>.4f} {omean=:>.4f} {ostd=:>.4f} {inorml=:.4f} {inormr=:.4f} {onorml=:.4f} {onormr=:.4f}")
            i += 1

        log.update({
            f"act/mean/{name}": x.mean().float().cpu().item(),
            f"act/std/{name}": x.std().float().cpu().item(),
            f"act/meanl/{name}": x[:, :seq_len//2].mean().float().cpu().item(),
            f"act/meanr/{name}": x[:, seq_len//2:].mean().float().cpu().item(),
            f"act/outmean/{name}": y.mean().float().cpu().item(),
            f"act/outstd/{name}": y.std().float().cpu().item(),
        })

    handles = []
    try:
        for name, p in model.named_modules():
            if any(inf in name for inf in infix):
                handles.append(p.register_forward_hook(partial(hook, name=name)))
        yield log
    finally:
        for handle in handles:
            handle.remove()


def print_weights(model, full=False):
    log = {}
    for n, p in model.named_parameters():
        # print weight scales of all parameters
        if not full and n.startswith('blocks') and 'blocks.0.' not in n:
            # just one block is fine: they are all the same at init
            continue
        if p.dim() == 2:
            s, c = torch.linalg.svdvals(p).round().int().unique(return_counts=True)
            s, c = s.tolist(), c.tolist()
        else:
            s, c = [], []

        log[f'weight/mean/{n}'] = p.mean().item()
        log[f'weight/std/{n}'] = p.std().item()
        log[f'weight/norm/{n}'] = p.norm().item()
        log[f'weight/singular/{n}'] = s[-1] if s else -1

        singular = f'singular {s} counts {c}' if s else ''
        print('weight', n, tuple(p.shape), f'norm {p.norm().item():.4f} mean {p.mean().item():.4f} std {p.std().item():.4f}', singular)
    return log


def summarize_gradients(model):
    log = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            log[f'grad/mean/{n}'] = p.grad.mean().float().cpu().item()
            log[f'grad/std/{n}'] = p.grad.std().float().cpu().item()
            log[f'grad/norm/{n}'] = p.grad.norm().float().cpu().item()

            print('grad', n, tuple(p.grad.shape), f'norm {p.grad.norm().item():.4f} mean {p.grad.mean().item():.4f} std {p.grad.std().item():.4f}')

    return log