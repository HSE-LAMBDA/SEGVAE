# TODO(julia): remove this file
import torch
import roboscientist.equation.operators as rs_operators

from scipy.optimize import minimize

import numpy as np


def _loss(constants, X, y, equation):
    y_hat = equation.func(X, constants)
    loss = (np.real((y_hat - y) ** 2)).mean()
    return np.abs(loss)


def optimize_constants(candidate_equation, X, y, method='bfgs', **kwargs):
    if method in METHODS:
        return METHODS[method](candidate_equation, X, y, **kwargs)
    else:
        raise NotImplementedError(f"Optimization method {method} do not implemented")


def bfgs_optimize(candidate_equation, X, y):
    if candidate_equation.const_count() > 0:
        success = False
        tries = 0
        constants = None
        while not success:
            if tries >= 5:
                break
            tries += 1
            res = minimize(lambda constants: _loss(constants, X, y, candidate_equation),
                np.random.uniform(low=0.1, high=1, size=candidate_equation.const_count()))
            success = res.success
            constants = res.x
        return constants
    return None


def adam_optimize(candidate_equation, X, y, learning_rate=1e-3, gtol=1e-2, max_iter=3_000, device='cpu'):
    n_const = candidate_equation.const_count()
    if n_const > 0:
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        constants = torch.randn(n_const, requires_grad=True, device=device)
        opt = torch.optim.Adam([constants], lr=learning_rate)
        for epoch in range(max_iter):
            y_pred = candidate_equation.func(X, constants)
            loss = torch.mean((y_pred - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if constants.grad.data.norm(2) < gtol:
                print('gtol')
                break
        return constants.detach().cpu().numpy()
    return None


METHODS = {
    'bfgs': bfgs_optimize,
    'adam': adam_optimize,
}
