# TODO(julia): delete or rewrite this file

import random
import numpy as np
from roboscientist.equation import operators


def generate_formula(all_tokens, max_len, functions):
    while True:
        const_ind = 0
        formula = []
        tokens_required = 1
        for _ in range(max_len):
            token = random.choice(all_tokens)
            if 'const' in token:
                token = token
                const_ind += 1
            if 'float' in token:
                token = random.choice(operators.FLOAT_CONST + operators.INT_CONST)
            formula.append(token)
            if token in functions:
                tokens_required += (operators.OPERATORS[token].arity - 1)
            else:
                tokens_required -= 1
            if tokens_required == 0:
                return ' '.join(formula)


def generate_pretrain_dataset(size, max_len, file=None, functions=None, all_tokens=None,
                              formula_predicate=None):
    if all_tokens is None:
        # all_tokens = ['x1', 'sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub', 'const']
        all_tokens = ['x1', 'sin', 'add', 'cos', 'mul',]
        # "Symbol('const%d')",
    if functions is None:
        # functions = ['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub']
        functions = ['sin', 'add', 'cos', 'mul']
    if formula_predicate is None:
        formula_predicate = lambda func: True

    formulas = []
    while len(formulas) < size:
        new_formulas = [generate_formula(all_tokens, max_len, functions) for _ in range(size)]
        # new_formulas = [formula_infix_utils.clear_redundant_operations(
        #     f.split(), functions, arities) for f in new_formulas]
        # new_formulas = [' '.join(f) for f in new_formulas]
        formulas += new_formulas
        formulas = list(np.unique(formulas))
        formulas = [formula for formula in formulas if formula_predicate(formula.strip().split())]
        # print(len(formulas))
        formulas = formulas[:size]

    if file is not None:
        with open(file, 'w') as f:
            f.write('\n'.join(formulas))
    return formulas


if __name__ == '__main__':
    generate_pretrain_dataset(20000, 14, 'train')
    generate_pretrain_dataset(10000, 14, 'val')
