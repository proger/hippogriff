"""
A Tape provides a sequence of batches.
One large training file is split into N uniform parts ("tapes"), where N is your batch size.
Every next sequence in the batch continues from the previous one.
"""
from itertools import islice
import numpy as np
import torch
import torch.nn.functional as F
from typing import Protocol


class IxSupervision(Protocol):
    "A protocol for a sequence of batches. Tensors have shape (N, T)."
    def __getitem__(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class Tape(IxSupervision):
    def __init__(self, data, batch_size, seq_len, seed=-1, device='cuda'):
        self.device = device
        self.data = data
        # there are batch_size tapes of length tape_len
        self.tape_len = len(data) // batch_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sequences = (self.tape_len + seq_len - 1) // seq_len
        self.iter = None
        # setting the seed turns the Tape into a regular iid sequence sampler
        self.generator = torch.Generator().manual_seed(seed) if seed >= 0 else None

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        yield from (self[i] for i in range(0, self.sequences))

    def __getitem__(self, step):
        if self.generator is not None:
            ix = torch.randint(len(self.data) - self.seq_len, (self.batch_size,), generator=self.generator).tolist()
        else:
            i = step % self.sequences
            ix = (torch.arange(0, self.batch_size) * self.tape_len + i * self.seq_len).tolist()
        #print('training batch offsets', ix)
        x = torch.stack([self._seq(i) for i in ix])
        y = torch.stack([self._seq(i+1, -100) for i in ix])
        x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x, y

    def _seq(self, i, padding=0):
        x = torch.from_numpy((self.data[i:i+self.seq_len]).astype(np.int64))
        return F.pad(x, (0, self.seq_len - x.shape[0]), value=padding)


class LanguiniTape(IxSupervision):
    "Languini provides a streaming interface for data, this object wraps it into a random access-like interface."
    def __init__(self, iterator, max_batches=-1):
        self.iterator = iterator
        self.step = -1
        self.generator = None # for compatibility with Tape
        self.max_batches = max_batches

    def __iter__(self):
        yield from islice(((x.squeeze(0).contiguous(), y.squeeze(0).contiguous()) for x, y, _ in self.iterator), 0, self.max_batches)

    def __getitem__(self, step):
        if step != self.step + 1:
            raise ValueError(f'steps over languini dataset must advance by 1. requested {step} but current step is {self.step}')
        if self.max_batches > 0 and step >= self.max_batches:
            raise IndexError(f'requested step {step} exceeds max_batches {self.max_batches}')
        x, y, _ = next(self.iterator)
        self.step = step
        return x.squeeze(0).contiguous(), y.squeeze(0).contiguous() # squeeze the micro batch dimension


class Tapes:
    choices = ["enwik8", "languini"]
    vocab_size: int
    seq_len: int
    train: IxSupervision
    valid: IxSupervision
    test: IxSupervision

    def __init__(self, vocab_size: int, seq_len: int, train: IxSupervision, valid: IxSupervision, test: IxSupervision | None = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.train = train
        self.valid = valid
        self.test = test or valid

    @classmethod
    def enwik8(cls, batch_size=32, seed=-1):
        return cls(
            vocab_size=205,
            seq_len=512,
            train=Tape(np.memmap('data/enwik8.train', dtype=np.uint8, mode='r'), batch_size=batch_size, seq_len=512, seed=seed),
            valid=Tape(np.memmap('data/enwik8.val', dtype=np.uint8, mode='r'), batch_size=128, seq_len=512),
            test=Tape(np.memmap('data/enwik8.test', dtype=np.uint8, mode='r'), batch_size=128, seq_len=512),
        )

    @classmethod
    def languini(cls, batch_size=32, **kwargs):
        """
        This tape provides access to training from the Languini Books dataset.
        data/books directory is assumed to be available.
        """
        try:
            from languini.dataset_lib.languini_books import LanguiniDatasetIterator
        except ImportError as e:
            raise ImportError("Install Languini Kitchen from: https://github.com/languini-kitchen/languini-kitchen") from e
        return cls(
            vocab_size=16384,
            seq_len=512,
            train=LanguiniTape(LanguiniDatasetIterator(
                data_path='data/books/books_16384',
                split='train',
                repeat=False,
                global_batch_size=batch_size,
                batch_idxs=range(batch_size),
                micro_batches=1,
                sequence_length=512,
                device='cuda',
                end_of_doc_token=2,
            )),
            valid=LanguiniTape(LanguiniDatasetIterator(
                data_path='data/books/books_16384',
                split='test',
                repeat=True,
                global_batch_size=32,
                batch_idxs=range(32),
                micro_batches=1,
                sequence_length=512,
                device='cuda',
                end_of_doc_token=2,
            ), max_batches=512)
        )
