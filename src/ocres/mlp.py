"""Small NumPy MLP used by the offline impression feasibility experiment."""
from __future__ import annotations

import numpy as np


class MLPClassifier:
    """One-hidden-layer classifier with deterministic full-batch Adam."""

    def __init__(self, input_dim, output_dim, hidden_dim=32, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def _forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        return h, logits

    def fit(self, x, y, epochs=800, lr=0.01, weight_decay=1e-4):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        counts = np.bincount(y, minlength=self.b2.size)
        present = counts > 0
        class_w = np.zeros_like(counts, dtype=np.float64)
        class_w[present] = n / (present.sum() * counts[present])
        sample_w = class_w[y]
        sample_w /= sample_w.mean()

        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, epochs + 1):
            h, logits = self._forward(x)
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            grad = prob
            grad[np.arange(n), y] -= 1
            grad *= sample_w[:, None] / n
            grads = [
                x.T @ ((grad @ self.w2.T) * (1 - h * h)) + weight_decay * self.w1,
                ((grad @ self.w2.T) * (1 - h * h)).sum(axis=0),
                h.T @ grad + weight_decay * self.w2,
                grad.sum(axis=0),
            ]
            for i, (param, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mh = m[i] / (1 - beta1**step)
                vh = v[i] / (1 - beta2**step)
                param -= lr * mh / (np.sqrt(vh) + eps)
        return self

    def predict(self, x):
        return self._forward(np.asarray(x, dtype=np.float64))[1].argmax(axis=1)
