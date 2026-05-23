"""Weight-initialization distributions compatible with the Nengo Distribution API."""

import numpy as np
import nengo.dists


class TruncatedNormal(nengo.dists.Distribution):
    """Samples from a normal distribution truncated to ±limit."""

    def __init__(self, mean=0.0, stddev=1.0, limit=None):
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.limit = limit if limit is not None else 2.0 * stddev

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        total = int(np.prod(shape))
        out = []
        while len(out) < total:
            need = total - len(out)
            s = rng.normal(self.mean, self.stddev, need * 4)
            s = s[np.abs(s - self.mean) <= self.limit]
            out.extend(s[:need].tolist())
        return np.array(out[:total], dtype=np.float32).reshape(shape)


class VarianceScaling(nengo.dists.Distribution):
    """Scales variance according to fan_in / fan_out / fan_avg.

    Parameters
    ----------
    scale : float
        Multiplicative scale factor applied before the fan division.
    mode : "fan_in" | "fan_out" | "fan_avg"
        Which fan quantity to divide by.
    distribution : "uniform" | "normal"
        Whether to sample from a (truncated) normal or a uniform distribution.
    """

    def __init__(self, scale=1.0, mode="fan_avg", distribution="uniform"):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        fan_in = d if d is not None else n
        fan_out = n

        scale = self.scale
        if self.mode == "fan_in":
            scale /= fan_in
        elif self.mode == "fan_out":
            scale /= fan_out
        else:  # fan_avg
            scale /= (fan_in + fan_out) / 2.0

        if self.distribution == "uniform":
            limit = np.sqrt(3.0 * scale)
            return rng.uniform(-limit, limit, shape).astype(np.float32)
        else:
            std = np.sqrt(scale)
            return TruncatedNormal(mean=0.0, stddev=std, limit=2.0 * std).sample(
                n, d, rng
            )


class Glorot(VarianceScaling):
    """Glorot / Xavier initialization: VarianceScaling with mode='fan_avg'."""

    def __init__(self, scale=1.0, distribution="uniform"):
        super().__init__(scale=scale, mode="fan_avg", distribution=distribution)


class He(VarianceScaling):
    """He / Kaiming initialization: VarianceScaling(scale²) with mode='fan_in'."""

    def __init__(self, scale=1.0, distribution="uniform"):
        # He uses scale² to match the convention in the original paper
        super().__init__(scale=scale ** 2, mode="fan_in", distribution=distribution)
