from tensor import Tensor
from typing import List, Dict
import numpy as np

class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")
        self.params = params
        self.lr = float(lr)
        self.mu = float(momentum)  
        self._vel: Dict[int, np.ndarray] = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            if self.mu > 0.0:
                key = id(p)
                v = self._vel.get(key)
                if v is None:
                    v = np.zeros_like(p.data)
                v = self.mu * v - self.lr * p.grad
                p.data += v
                self._vel[key] = v
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)

class Adam:
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self._m1: Dict[int, np.ndarray] = {}
        self._m2: Dict[int, np.ndarray] = {}
        self._step = 0

    def step(self):
        self._step += 1
        b1_correction = 1.0 - self.beta1 ** self._step
        b2_correction = 1.0 - self.beta2 ** self._step

        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            m1 = self._m1.get(pid)
            m2 = self._m2.get(pid)
            if m1 is None:
                m1 = np.zeros_like(p.data)
            if m2 is None:
                m2 = np.zeros_like(p.data)

            m1 = self.beta1 * m1 + (1.0 - self.beta1) * p.grad
            m2 = self.beta2 * m2 + (1.0 - self.beta2) * (p.grad * p.grad)

            m1_hat = m1 / b1_correction
            m2_hat = m2 / b2_correction

            p.data -= self.lr * (m1_hat / (np.sqrt(m2_hat) + self.eps))

            self._m1[pid] = m1
            self._m2[pid] = m2

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)
