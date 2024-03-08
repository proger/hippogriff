import numpy as np
import torch
import torch.nn.functional as F


class Tapes:
    @classmethod
    def enwik8(cls, args):
        x = cls()
        x.vocab_size = 205
        x.seq_len = 512
        x.train = Tape(np.memmap('data/enwik8.train', dtype=np.uint8, mode='r'), batch_size=args.batch_size, seq_len=x.seq_len, seed=args.seed)
        x.valid = Tape(np.memmap('data/enwik8.val', dtype=np.uint8, mode='r'), batch_size=128, seq_len=x.seq_len)
        x.test = Tape(np.memmap('data/enwik8.test', dtype=np.uint8, mode='r'), batch_size=128, seq_len=x.seq_len)
        return x


class Tape:
    def __init__(self, data, batch_size, seq_len, seed=-1, device='cuda'):
        self.device = device
        self.data = data
        # there are batch_size tapes of length tape_len
        self.tape_len = len(data) // batch_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.sequences = (self.tape_len + seq_len - 1) // seq_len
        self.iter = None
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

