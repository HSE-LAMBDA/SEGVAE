import roboscientist.equation.equation as rs_equation

import numpy as np

EQUATIONS = {
    'N1': (['add', 'x1', 'add', 'mul', 'x1', 'x1', 'mul', 'mul', 'x1', 'x1', 'x1'], ['x1']),
    'N2': (['mul', 'add', 'x1', '1.0', 'mul', 'x1', 'add', '1.0', 'mul', 'x1', 'x1'], ['x1']),
    'N3': (['add', 'x1', 'mul', 'add', 'x1', 'mul', 'x1', 'x1', 'add', 'x1', 'mul', 'mul', 'x1', 'x1', 'x1'], ['x1']),
    'N4': (['add', 'x1', 'add', 'mul', 'x1', 'x1', 'add', 'pow', 'x1', '3.0',
          'add', 'pow', 'x1', '4.0', 'add', 'pow', 'x1', '5.0', 'pow', 'x1', '6.0'], ['x1']),
    'N5': (['sub', 'mul', 'sin', 'mul', 'x1', 'x1', 'cos', 'x1', '1.0'], ['x1']),
    'N6': (['add', 'sin', 'x1', 'sin', 'add', 'x1', 'mul', 'x1', 'x1'], ['x1']),
    'N7': (['add', 'ln', 'add', 'x1', '1.0', 'ln', 'add', 'mul', 'x1', 'x1', '1.0'], ['x1']),
    'N8': (['pow', 'x1', 'div', '1.0', '2.0'], ['x1']),
    'N9': (['add', 'sin', 'x1', 'sin', 'mul', 'x2', 'x2'], ['x1', 'x2']),
    'N10': (['mul', '2.0', 'mul', 'sin', 'x1', 'cos', 'x2'], ['x1', 'x2']),
    'N11': (['pow', 'x1', 'x2'], ['x1', 'x2']),
    'N12': (['add', 'sub', 'pow', 'x1', '4.0', 'pow', 'x1', '3.0',
           'sub', 'div', 'mul', 'x2', 'x2', '2.0', 'x2'], ['x1', 'x2']),
    'L5': (['add', 'sub', 'pow', 'x1', '4.0', 'pow', 'x1', '3.0',
            'sub', 'mul', 'x1', 'x1', 'x2'], ['x1', 'x2']),
    'L7': (['div', 'sub', 'exp', 'x1', 'exp', 'mul', '-1.0', 'x1', '2.0'], ['x1']),
    'L8': (['div', 'add', 'exp', 'x1', 'exp', 'mul', '-1.0', 'x1', '2.0'], ['x1']),
    'L10': (['mul', 'mul', '6.0', 'sin', 'x1', 'cos', 'x2'], ['x1', 'x2']),
    'L17': (['mul', 'mul', '4.0', 'sin', 'x1', 'cos', 'x2'], ['x1', 'x2']),
    'L22': (['exp', 'mul', 'mul', 'x1', 'x1', 'div', '-1.0', '2.0'], ['x1']),

    'NEAT8': (['div', 'exp', 'mul', '-1.0', 'pow2', 'sub', 'x1', '1.0',
               'add', 'div', '6.0', '5.0', 'pow2', 'sub', 'x2', 'div', '5.0', '2.0'], ['x1', 'x2']),
    'NEAT9': (['add', 'div', '1.0', 'add', '1.0', 'pow', 'x1', 'mul', '-1.0', '4.0',
               'div', '1.0', 'add', '1.0', 'pow', 'x2', 'mul', '-1.0', '4.0'],
              ['x1', 'x2']),
    'R2': (['div', 'add', 'sub', 'pow', 'x1', '5.0', 'mul', '3.0', 'pow', 'x1', '3.0', '1.0',
            'add', 'mul', 'x1', 'x1', '1.0'], ['x1']),
    'R3': (['div', 'add', 'pow', 'x1', '6.0', 'pow', 'x1', '5.0', 'add', 'add', 'add', 'add',
            'pow', 'x1', '4.0', 'pow', 'x1', '3.0', 'pow', 'x1', '2.0', 'x1', '1.0'], ['x1']),
}


if __name__ == '__main__':
    eq = rs_equation.Equation(['add', 'sin', 'x1', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '(sin(x1) + x1)'
    assert np.allclose(eq.func(X = np.array([[1], [0]])), np.array([1.84147098, 0.]))

    eq = rs_equation.Equation(['add', 'add', 'x1', 'x1', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '((x1 + x1) + x1)'
    assert np.allclose(eq.func(X=np.array([[1], [0]])), np.array([3., 0.]))

    eq = rs_equation.Equation(['add', 'add', 'x1', 'x2', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '((x1 + x2) + x1)'
    assert np.allclose(eq.func(X=np.array([[1., 3], [0, 5]])), np.array([5., 5.]))
