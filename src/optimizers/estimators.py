from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np


class BaseEstimator(ABC):
    """
    Base interface for per-link capacity estimators used by No-CSI optimizers.
    """

    @abstractmethod
    def estimate(self, t: int) -> np.ndarray:
        """
        Returns the per-link capacity estimate c_hat(t).
        """

    @abstractmethod
    def update(self, c_t: np.ndarray) -> None:
        """
        Updates estimator state after observing realized capacities c_t.
        """


@dataclass
class MeanEstimator(BaseEstimator):
    """
    Per-link estimator:
        c_hat = max(eps, mean(history))

    Warmup: for the first `warmup` slots, returns constant `warmup_value` since
    no history exists at decision time.
    """

    num_links: int
    warmup_value: float
    warmup: int = 5
    eps: float = 1e-6
    _hist: List[List[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hist = [[] for _ in range(self.num_links)]

    def estimate(self, t: int) -> np.ndarray:
        if t < self.warmup:
            return np.full(self.num_links, float(self.warmup_value), dtype=float)

        c_hat = np.zeros(self.num_links, dtype=float)
        for i in range(self.num_links):
            xs = np.array(self._hist[i], dtype=float)
            if xs.size == 0:
                c_hat[i] = float(self.warmup_value)
                continue
            c_hat[i] = max(self.eps, float(xs.mean()))
        return c_hat

    def update(self, c_t: np.ndarray) -> None:
        if c_t.shape[0] != self.num_links:
            raise ValueError(f"Expected c_t length {self.num_links}, got {c_t.shape[0]}.")
        for i in range(self.num_links):
            self._hist[i].append(float(c_t[i]))


@dataclass
class MeanMinusZStdLCB(BaseEstimator):
    """
    Per-link lower-confidence estimator:
        c_hat = max(eps, mean - z * std)

    Warmup: for the first `warmup` slots, returns a constant conservative guess
    (c_min) since no history exists at decision time.
    """

    num_links: int
    warmup_value: float = 1e12
    z: float = 1.2815515655446004  # ~ one-sided 90% normal quantile
    warmup: int = 5
    eps: float = 1e-6
    _hist: List[List[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hist = [[] for _ in range(self.num_links)]

    def estimate(self, t: int) -> np.ndarray:
        if t < self.warmup:
            return np.full(self.num_links, float(self.warmup_value), dtype=float)

        c_hat = np.zeros(self.num_links, dtype=float)
        for i in range(self.num_links):
            xs = np.array(self._hist[i], dtype=float)
            if xs.size == 0:
                c_hat[i] = float(self.warmup_value)
                continue
            mu = float(xs.mean())
            sigma = float(xs.std(ddof=1)) if xs.size >= 2 else 0.0
            c_hat[i] = max(self.eps, mu - self.z * sigma)
        return c_hat

    def update(self, c_t: np.ndarray) -> None:
        if c_t.shape[0] != self.num_links:
            raise ValueError(f"Expected c_t length {self.num_links}, got {c_t.shape[0]}.")
        for i in range(self.num_links):
            self._hist[i].append(float(c_t[i]))


@dataclass
class LastObservationEstimator(BaseEstimator):
    """
    Per-link estimator:
        c_hat_i(t) = c_i(t-1)  (last observed value)

    Warmup:
        If no observation exists yet, returns a large `warmup_value` to
        effectively produce `eta=1` under the closed-form mapping used by
        the online baselines.
    """

    num_links: int
    warmup_value: float = 1e12
    _last_c: np.ndarray | None = field(default=None, init=False, repr=False)

    def estimate(self, t: int) -> np.ndarray:  # noqa: ARG002 - t kept for interface consistency
        if self._last_c is None:
            return np.full(self.num_links, float(self.warmup_value), dtype=float)
        return np.array(self._last_c, copy=True)

    def update(self, c_t: np.ndarray) -> None:
        if c_t.shape[0] != self.num_links:
            raise ValueError(f"Expected c_t length {self.num_links}, got {c_t.shape[0]}.")
        self._last_c = np.array(c_t, copy=True, dtype=float)


@dataclass
class RunningMinEstimator(BaseEstimator):
    """
    Per-link estimator:
        c_hat_i(t) = min_{tau < t} c_i(tau)

    Warmup:
        If no observation exists yet, returns a large `warmup_value` to
        effectively produce `eta=1` under the closed-form mapping used by
        the online baselines.
    """

    num_links: int
    warmup_value: float = 1e12
    _min_c: np.ndarray | None = field(default=None, init=False, repr=False)

    def estimate(self, t: int) -> np.ndarray:  # noqa: ARG002 - t kept for interface consistency
        if self._min_c is None:
            return np.full(self.num_links, float(self.warmup_value), dtype=float)
        return np.array(self._min_c, copy=True)

    def update(self, c_t: np.ndarray) -> None:
        if c_t.shape[0] != self.num_links:
            raise ValueError(f"Expected c_t length {self.num_links}, got {c_t.shape[0]}.")
        c_t = np.array(c_t, copy=True, dtype=float)
        if self._min_c is None:
            self._min_c = c_t
        else:
            self._min_c = np.minimum(self._min_c, c_t)


@dataclass
class MovingAverageEstimator(BaseEstimator):
    """
    Per-link sliding-window moving average estimator:
        c_hat_i(t) = mean of the last `window` observations of link i.

    Warmup:
        Until the first observation exists, returns a large `warmup_value` to
        effectively produce `eta=1` under the closed-form mapping used by
        the online baselines.
    """

    num_links: int
    window: int = 10
    warmup_value: float = 1e12
    _hist: List[deque] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be >= 1, got {self.window}.")
        self._hist = [deque(maxlen=self.window) for _ in range(self.num_links)]

    def estimate(self, t: int) -> np.ndarray:  # noqa: ARG002 - t kept for interface consistency
        c_hat = np.zeros(self.num_links, dtype=float)
        for i in range(self.num_links):
            if len(self._hist[i]) == 0:
                c_hat[i] = float(self.warmup_value)
            else:
                c_hat[i] = float(np.mean(np.fromiter(self._hist[i], dtype=float)))
        return c_hat

    def update(self, c_t: np.ndarray) -> None:
        if c_t.shape[0] != self.num_links:
            raise ValueError(f"Expected c_t length {self.num_links}, got {c_t.shape[0]}.")
        for i in range(self.num_links):
            self._hist[i].append(float(c_t[i]))
