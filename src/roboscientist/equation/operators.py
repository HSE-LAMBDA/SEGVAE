import numpy as np
import sympy as sp
import torch


class Operator:
    def __init__(self, func, name, repr, arity, complexity, sympy):
        self.func = func
        self.name = name
        self.repr = repr
        self.arity = arity
        self.complexity = complexity
        self.sympy = sympy


def _SAFE_LOG_FUNC(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0.0001, torch.log(torch.abs(x)), torch.tensor(0.0))
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0.0001, np.log(np.abs(x)), np.nan)


def _SAFE_DIV_FUNC(x, y):
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        return torch.where(torch.abs(y) > 0.001, torch.divide(x, y), torch.tensor(0.0))
    else:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return np.where(np.abs(y) > 0.001, np.divide(x, y), np.nan)


def _SAFE_SQRT_FUNC(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, torch.sqrt(torch.abs(x)), torch.tensor(0.0))
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0, np.sqrt(np.abs(x)), np.nan)


def _SAFE_EXP_FUNC(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x < 10, torch.exp(x), torch.exp(torch.tensor(10.0)))
    else:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return np.where(x < 10, np.exp(x), np.exp(10))


def _SAFE_POW_FUNC(x, y):
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        coeff = torch.where(torch.eq(torch.fmod(y, 1), 0), (-1) ** y, 0.0)
        return torch.where(x > 0, _SAFE_EXP_FUNC(y * _SAFE_LOG_FUNC(x)),
                           coeff * _SAFE_EXP_FUNC(y * _SAFE_LOG_FUNC(torch.abs(x))))
    else:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            coeff = np.where(np.equal(np.mod(y, 1), 0), (-1) ** y, 0.0).astype(np.float64)
            return np.where(x > 0, _SAFE_EXP_FUNC(y * _SAFE_LOG_FUNC(x)),
                            coeff * _SAFE_EXP_FUNC(y * _SAFE_LOG_FUNC(np.abs(x))))


OPERATORS = {
    'add': Operator(
        func=lambda x, y: x + y,
        name='add',
        repr=lambda x, y: f'({x} + {y})',
        arity=2,
        complexity=1,
        sympy=lambda x, y: sp.Add(x, y),
    ),
    'sub': Operator(
        func=lambda x, y: x - y,
        name='sub',
        repr=lambda x, y: f'({x} - {y})',
        arity=2,
        complexity=1,
        sympy=lambda x, y: sp.Add(x, sp.Mul(-1, y)),
    ),
    'mul': Operator(
        func=lambda x, y: x * y,
        name='mul',
        repr=lambda x, y: f'({x} * {y})',
        arity=2,
        complexity=1,
        sympy=lambda x, y: sp.Mul(x, y),
    ),
    'sin': Operator(
        func=lambda x: np.sin(x),
        name='sin',
        repr=lambda x: f'sin({x})',
        arity=1,
        complexity=3,
        sympy=lambda x: sp.sin(x),
    ),
    'cos': Operator(
        func=lambda x: np.cos(x),
        name='cos',
        repr=lambda x: f'cos({x})',
        arity=1,
        complexity=3,
        sympy=lambda x: sp.cos(x),
    ),
    'log': Operator(
        func=lambda x: _SAFE_LOG_FUNC(x),
        name='safe_log',
        repr=lambda x: f'log({x})',
        arity=1,
        complexity=4,
        sympy=lambda x: sp.log(x, evaluate=False),
    ),
    'sqrt': Operator(
        func=lambda x: _SAFE_SQRT_FUNC(x),
        name='safe_sqrt',
        repr=lambda x: f'sqrt({x})',
        arity=1,
        complexity=2,
        sympy=lambda x: sp.sqrt(x, evaluate=False),
    ),
    'div': Operator(
        func=lambda x, y: _SAFE_DIV_FUNC(x, y),
        name='safe_div',
        repr=lambda x, y: f'({x} / {y})',
        arity=2,
        complexity=2,
        sympy=lambda x, y: sp.Mul(x, sp.Pow(y, -1)),
    ),
    'exp': Operator(
        func=lambda x: _SAFE_EXP_FUNC(x),
        name='safe_exp',
        repr=lambda x: f'(e^{x})',
        arity=1,
        complexity=4,
        sympy=lambda x: sp.exp(x, evaluate=False),
    ),
    'pow': Operator(
        func=lambda x, y: _SAFE_POW_FUNC(x, y),
        name='safe_pow',
        repr=lambda x, y: f'({x}^{y})',
        arity=2,
        complexity=4,
        sympy=lambda x, y: sp.Pow(x, y, evaluate=False),
    ),
    'pow2': Operator(
        func=lambda x: _SAFE_POW_FUNC(x, 2),
        name='safe_pow2',
        repr=lambda x: f'({x}^{2})',
        arity=1,
        complexity=3,
        sympy=lambda x: sp.Pow(x, 2, evaluate=False),
    ),
    'e': Operator(
        func=lambda: np.e,
        name='e',
        repr=lambda: f'e',
        arity=0,
        complexity=1,
        sympy=lambda: sp.E,
    ),
    'pi': Operator(
        func=lambda: np.e,
        name='pi',
        repr=lambda: f'pi',
        arity=0,
        complexity=1,
        sympy=lambda: sp.pi,
    ),
    '0.5': Operator(
        func=lambda: 0.5,
        name='half',
        repr=lambda: f'0.5',
        arity=0,
        complexity=1,
        sympy=lambda: 0.5,
    ),
}


VARIABLES = {
    'x1': 0,
    'x2': 1,
    'x3': 2,
    'x4': 3,
    'x5': 4
}
CONST_SYMBOL = 'const'

INT_CONST = ['-1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
FLOAT_CONST = []

VAR_CONST_COMPLEXITY = 1

if __name__ == '__main__':
    print(_SAFE_LOG_FUNC(np.array([0, 1, -1, 3])))
