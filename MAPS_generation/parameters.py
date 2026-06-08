import itertools

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube


class ParameterSpace:
    def __init__(self, dict_params):
        self.dict_params = dict(sorted(dict_params.items()))
        self.param_names = list(self.dict_params.keys())
        self.dim = len(self.param_names)

        self.key2idx = {k: i for i, k in enumerate(self.param_names)}
        self.lb = np.zeros(self.dim, dtype=float)
        self.ub = np.ones(self.dim, dtype=float)

    def encode(self, params):
        x = np.empty(self.dim, dtype=float)
        for k, spec in self.dict_params.items():
            low, high = spec['range']
            i = self.key2idx[k]
            x[i] = (params[k] - low) / (high - low)
        return x

    def decode(self, x, clamp=True):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f'x must have shape ({self.dim},), got {x.shape}')

        if clamp:
            x = np.clip(x, self.lb, self.ub)

        params = {}
        for k, spec in self.dict_params.items():
            low, high = spec['range']
            i = self.key2idx[k]
            params[k] = low + x[i] * (high - low)
        return params

    def sample(self):
        params = {}
        for k, spec in self.dict_params.items():
            low, high = spec['range']
            params[k] = np.random.uniform(low, high)
        return params

    def bounds(self):
        return self.lb.copy(), self.ub.copy()

    def __str__(self):
        lines = [
            'ParameterSpace',
            f'  Total dimension: {self.dim}',
            '  Parameters:',
        ]
        for k in self.param_names:
            low, high = self.dict_params[k]['range']
            lines.append(f'    - {k}: [{low}, {high}]')
        return '\n'.join(lines)


class ParameterSweep:
    def __init__(self, dict_space, dict_sweep):
        self.dict_space = dict_space
        self.dict_sweep = dict_sweep
        assert set(dict_sweep).issubset(dict_space)

        self.grids = self.make_grid()
        self.keys = list(self.grids.keys())
        self.num_images = np.prod([len(v) for v in self.grids.values()])

    def make_grid(self):
        grids = {}
        for name, num in self.dict_sweep.items():
            values = self.dict_space[name]
            low, high = values['range']
            if values.get('circular', False):
                grids[name] = np.linspace(low, high, num, endpoint=False)
            else:
                grids[name] = np.linspace(low, high, num)

        return grids

    def __iter__(self):
        for values in itertools.product(*self.grids.values()):
            params = dict(zip(self.keys, values))
            yield params

    def __repr__(self):
        return f'ParameterSweep({self.dict_sweep})'

    def __len__(self):
        return self.num_images

    def to_dataframe(self):
        return pd.DataFrame(self)


class ParameterSampler:
    def __init__(self, parameter_space, num_images, seed=None):
        self.space = parameter_space
        self.num_images = num_images
        self.seed = seed
        self.keys = list(self.space.dict_params)
        self.sample = LatinHypercube(d=len(self.keys), seed=self.seed).random(n=self.num_images)

    def sample_to_theta(self, u_row) -> dict:
        theta = {}
        for j, k in enumerate(self.keys):
            spec = self.space.dict_params[k]
            low, high = spec["range"]
            u = float(u_row[j])
            theta[k] = low + u * (high - low)
        return theta

    def __iter__(self):
        for i in range(self.num_images):
            yield self.sample_to_theta(self.sample[i])

    def __len__(self):
        return self.num_images

    def to_dataframe(self):
        return pd.DataFrame(self)