import numpy as np
import scipy as sp
import pandas as pd


class Analyzer():
    def __init__(self, *args, **kwargs):
        self.options = {
            'precision': np.float64,
            'digits': 6,
        }

    def change_options(self, precision=None, digits=None):
        if precision is not None: self.options['precision'] = precision
        if digits is not None: self.options['digits'] = digits

    def array(self, data, dtype=None):
        if dtype is None: dtype = self.options['precision']
        try: return np.array(data, dtype=dtype)
        except: return np.array([data], dtype=dtype)

    def empty(self, shape, value=None, dtype=None):
        if dtype is None: dtype = self.options['precision']
        if value is None: return np.empty(shape, dtype=dtype)
        else: return np.full(shape, value, dtype=dtype)

    def numeric(self, data, nan=None, posinf=None, neginf=None):
        return np.nan_to_num(data, nan, posinf, neginf)

    def shape(self, data, axis=None):
        if axis is None: return np.shape(data)
        else: return np.shape(data)[axis]

    def size(self, data, axis=None):
        return np.size(data, axis=axis)

    def transpose(self, data):
        return np.transpose(data)

    def move(self, data, axis=None):
        return np.moveaxis(data, *axis)

    def swap(self, data, axis=None):
        return np.swapaxes(data, *axis)

    def concat(self, data, axis=None):
        try: return np.concatenate(data, axis=axis)
        except: return np.concatenate([data], axis=axis)

    def stack(self, data, axis=None):
        try: return np.stack(data, axis=axis)
        except: return np.stack([data], axis=axis)

    def round(self, data):
        return np.round(data, self.options['digits'])

    def unique(self, data, axis=None):
        return np.unique(data, axis=axis)

    def count(self, data, values, axis=None):
        return np.array([np.sum(data == value, axis=axis) for value in values])

    def sort(self, data, axis=None, indices=False):
        if indices: return np.argsort(data, axis=axis)
        else: return np.sort(data, axis=axis)

    def max(self, data, axis=None, indices=False):
        if indices: return np.argmax(data, axis=axis)
        else: return np.max(data, axis=axis)

    def min(self, data, axis=None, indices=False):
        if indices: return np.argmin(data, axis=axis)
        else: return np.min(data, axis=axis)

    def median(self, data, axis=None):
        return np.median(data, axis=axis)

    def mean(self, data, axis=None):
        return np.mean(data, axis=axis)

    def reverse(self, data, axis=None):
        return np.flip(data, axis=axis)

    def flatten(self, data):
        return np.ravel(data)

    def reshape(self, data, shape):
        return np.reshape(data, shape)

    def expand(self, data, axis=None):
        return np.expand_dims(data, axis=axis)

    def squeeze(self, data, axis=None):
        return np.squeeze(data, axis=axis)

    def repeat(self, data, shape, axis=None):
        return np.repeat(data, shape, axis=axis)

    def tile(self, data, shape):
        return np.tile(data, shape)

    def split(self, data, shape, axis=None):
        return np.split(data, shape, axis=axis)

    def partition(self, data, shape, axis=None):
        return np.partition(data, shape, axis=axis)

    def crop(self, data):
        return np.crop(data)

    def pad(self, data, shape):
        return np.pad(data, shape)

    def accumulate(self, data, axis=None, mode='sum'):
        if mode == 'sum': return np.cumsum(data, axis=axis)
        elif mode == 'prod': return np.cumprod(data, axis=axis)

    def clip(self, data, lower=0, upper=1):
        return np.clip(data, lower, upper)

    def normalize(self, data, lower=0, upper=1):
        return np.add(lower, np.multiply(np.subtract(upper, lower), np.divide(np.subtract(data, np.min(data)), np.subtract(np.max(data), np.min(data)))))

    def n_average(self, data, n):
        # FIXME: use padding instead cropping
        return np.average(np.reshape(data[:(len(data) // n) * n], (-1, n)), axis=1)

    def range(self, data):
        return np.arange(*data)

    def space(self, data):
        return np.linspace(*data)

    def grid(self, data):
        return np.meshgrid(*data)

    def stats(self, data):
        # bool to -1 and +1, None to 0
        # nan only to fill
        data = data.astype(self.options['precision'])
        values, counts = np.unique(data, return_counts=True, axis=None)

        return {
            'count': len(data),
            'uniq': len(values),
            'nan': len(data[np.isnan(data)]),
            'inf': len(data[np.isinf(data)]),
            'zero': len(data[data == 0]),
            'pos': len(data[data > 0]),
            'neg': len(data[data < 0]),

            'min': self.round(np.min(values, axis=None)),
            'max': self.round(np.max(values, axis=None)),
            'ptp': self.round(np.ptp(values, axis=None)),
            'q1': self.round(np.quantile(data, .25, axis=None)),
            'q3': self.round(np.quantile(data, .75, axis=None)),
            'iqr': self.round(*np.diff(np.quantile(data, [.25, .75], axis=None))),
            'rare': self.round(values[np.argmin(counts)]),
            'mode': self.round(values[np.argmax(counts)]),
            'median': self.round(np.median(data, axis=None)),
            'mean': self.round(np.mean(data, axis=None)),
            'var': self.round(np.var(data, axis=None)),
            'std': self.round(np.std(data, axis=None)),
            'sum': self.round(np.sum(data, axis=None)),
            'prod': self.round(np.prod(data, axis=None)),
        }
