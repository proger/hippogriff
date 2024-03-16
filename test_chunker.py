"""
https://people.idsia.ch/~juergen/FKI-148-91ocr.pdf
"""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console

from align_utf8 import align_utf8_bytes_and_characters
from train_diagnostics import summarize_activations
from train_init import make_model


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter): pass
parser = argparse.ArgumentParser("Neural Sequence Chunker. Segment the input into chunks using a pretrained model. Every continuation of the chunk is predictable from its head.", formatter_class=Formatter)
parser.add_argument('--init', required=True, type=Path, help="load model weights from this checkpoint")
parser.add_argument('sentences', type=Path, help="File with sentences, one per line.")

device = 'cuda' # use CUDA_VISIBLE_DEVICES to choose the device until accelerated-scan supports cuda:N
dtype = torch.bfloat16
console = Console()


def align_utf8_bytes_and_characters(input_string: str) -> list[tuple[bytes, str]]:
    """
    Given a string привіт produce python code that aligns a sequence of utf-8 bytes for that string and its characters. Remember the alignment between unicode bytes and characters is non-uniform.
    """

    # Encode the input string to UTF-8
    utf8_bytes = input_string.encode('utf-8')

    # Align bytes and characters
    aligned_result = []
    byte_index = 0

    while byte_index < len(utf8_bytes):
        # Determine the number of bytes for the current character
        byte = utf8_bytes[byte_index]
        if byte & 0b10000000 == 0b00000000:
            char_size = 1
        elif byte & 0b11100000 == 0b11000000:
            char_size = 2
        elif byte & 0b11110000 == 0b11100000:
            char_size = 3
        elif byte & 0b11111000 == 0b11110000:
            char_size = 4
        else:
            raise ValueError("Invalid UTF-8 byte sequence")

        # Extract the character and update the byte index
        character = utf8_bytes[byte_index:byte_index + char_size].decode('utf-8')
        aligned_result.append((utf8_bytes[byte_index:byte_index + char_size], character))
        byte_index += char_size

    return aligned_result


@torch.inference_mode()
def evaluate(model, batches, diag_prefix='eval') -> tuple[float, dict]:
    model.eval()
    losses = []
    accuracy_sum, accuracy_count = 0, 0
    diag = {}
    for i, (input_ids, targets, length) in enumerate(batches):
        with summarize_activations(model, infix=['input', 'output'], verbose=i==0) as batch_diag:
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                logits = model(input_ids)
                targets = targets[:, :length]
                logits = logits[:, :length]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).mean()

            guess = logits.argmax(dim=-1).eq(targets)[0]
            guess = guess.cpu().numpy()[:length]
            accuracy_sum, accuracy_count = guess.sum().item(), targets.numel()
            og_bytes = b''.join([bytes([x]) for x in input_ids[0].cpu().numpy()[:length]])
            og_string = og_bytes.decode('utf-8')

            ali = align_utf8_bytes_and_characters(og_string)
            b0, s0 = ali[0]
            t = len(b0) - 1
            g = ' ' if guess[:t].all() else '.'
            console.print(i,f'{accuracy_sum / accuracy_count:.3f}', end=' ')
            console.print(s0, end='', style='cyan')
            cyan = False
            for b, s in ali[1:]:
                predictable = guess[t:t+len(b)].all()
                g += ' ' if predictable else '.'
                t += len(b)
                console.print(s, end='', style='cyan' if cyan else 'magenta')
                if not predictable:
                    cyan = not cyan
            console.print('')

            print(i, 'start', g, file=sys.stderr)

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


if __name__ == '__main__':
    args = parser.parse_args()

    vocab_size = 256
    def mktape():
        for line in args.sentences.open('rb'):
            input_ids, targets = line[:-1], line[1:]
            input_ids = torch.tensor(np.frombuffer(input_ids, dtype=np.uint8))
            targets = torch.tensor(np.frombuffer(targets, dtype=np.uint8))
            input_ids = input_ids.to(device).long()[None, :]
            targets = targets.to(device).long()[None, :]
            length = input_ids.shape[1]
            # pad with zeros to power of 2
            pad = 2**int(np.ceil(np.log2(max(32,input_ids.shape[1])))) - input_ids.shape[1]
            input_ids = F.pad(input_ids, (0, pad), value=0)
            targets = F.pad(targets, (0, pad), value=0)
            yield input_ids, targets, length

    tape = mktape()
            
    model = make_model(vocab_size, init=args.init)
    test = evaluate(model, tape)
    print(test)
