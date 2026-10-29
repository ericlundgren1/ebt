import numpy as np


class Loader():
    '''
    Custom dataloader because calling __getitem__ on each row gets really expensive on large datasets and large number of epochs
    '''
    def __init__(self, x, y, batch_size=128, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.i = 0

        if shuffle:
            ix = np.random.choice(np.arange(x.shape[0]), size=x.shape[0], replace=False)
            self.x = self.x[ix]
            self.y = self.y[ix]

    def __len__(self):
        whole_batches = self.x.shape[0] // self.batch_size
        rest = self.x.shape[0] % self.batch_size
        num_batches = whole_batches + (0 if rest == 0 else 1)
        return num_batches

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            ix_start = self.i * self.batch_size
            ix_end = ix_start + self.batch_size
            self.i += 1
            return (self.x[ix_start:ix_end], self.y[ix_start:ix_end]) if self.y is not None else self.x[ix_start:ix_end]
        else:
            raise StopIteration
        