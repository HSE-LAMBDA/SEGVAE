import os

import numpy as np
import sympy as sp
import pandas as pd
import argparse
from experiments import run_experiment
import roboscientist.equation.equation as rs_equation
import roboscientist.equation.operators as rs_operators
from equation_test import EQUATIONS


def get_offsprings(list_of_tokens, idx):
    if list_of_tokens[idx] in rs_operators.OPERATORS:
        open_nodes = rs_operators.OPERATORS[list_of_tokens[idx]].arity
    else:
        open_nodes = 0
    traversal = []
    for i, token in enumerate(list_of_tokens[idx + 1:]):
        if open_nodes == 0:
            break
        traversal.append(token)
        if token in rs_operators.OPERATORS:
            operator = rs_operators.OPERATORS[token]
            open_nodes += operator.arity - 1
        else:
            open_nodes -= 1

    return traversal


def predicate(list_of_tokens):
    for i, token in enumerate(list_of_tokens):
        offsprings = get_offsprings(list_of_tokens, i)
        if token == 'sin' or token == 'cos':
            if 'sin' in offsprings or 'cos' in offsprings:
                return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument("--eq", type=str)
    parser.add_argument("--noise", type=str)
    parser.add_argument("--latent", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--pretrain", type=int)
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    filename = f'{args.eq}_noise0.{args.noise:0>2}.csv'
    true_func, free_variables = EQUATIONS[args.eq]
    df = pd.read_csv(os.path.join(args.path, filename), sep=',', header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(X.shape)

    x_lows = [X.min(axis=0)[var_idx] for var_idx in range(X.shape[-1])]
    x_highs = [X.max(axis=0)[var_idx] for var_idx in range(X.shape[-1])]
    y_dom = (-np.inf, np.inf)
    domains_grid = (x_lows, x_highs, y_dom)

    run_experiment(
        X.values, y.values,
        functions=list(rs_operators.OPERATORS.keys()),
        free_variables=free_variables,
        wandb_proj='WANDB-PROJECT',
        project_name='EXP-NAME',
        constants=[],
        float_constants=rs_operators.FLOAT_CONST + rs_operators.INT_CONST,
        epochs=10,
        n_formulas_to_sample=5000,
        max_formula_length=30,
        formula_predicate=predicate,
        true_formula=rs_equation.Equation(true_func),
        latent=args.latent,
        lstm_hidden_dim=args.hidden,
        device=args.device,
        log_intermediate_steps=True,
        pretrain_path=args.pretrain,
        domains=domains_grid,
        simplification=True,
    )
