import roboscientist.equation.operators as rs_operators

from collections import deque
import numpy as np
import sympy as sp

class ConstantsCountError(Exception):
    pass


class InvalidEquationError(Exception):
    pass


class Equation:
    def __init__(self, prefix_list):
        self._prefix_list = prefix_list
        self._repr = None
        self._const_count = None
        self._status, self.complexity = self.validate()

    def check_validity(self):
        return self._status == 'OK', self._status

    def __len__(self):
        return len(self._prefix_list)

    def repr(self, constants=None):
        if constants is None:
            return self._repr
        stack = deque()
        const_ind = 0
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.VARIABLES or elem in rs_operators.FLOAT_CONST or elem in rs_operators.INT_CONST:
                stack.append(elem)
                continue
            if elem == rs_operators.CONST_SYMBOL:
                if constants is not None and const_ind < len(constants):
                    stack.append(str(constants[const_ind]))
                    const_ind += 1
                else:
                    raise ConstantsCountError(f'not enough constants passed {self._prefix_list}, {constants}')
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    return f'Invalid Equation {self._prefix_list}'
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.repr(*args))
                continue
            return f'Invalid symbol in Equation {self._prefix_list}'
        if len(stack) != 1:
            return f'Invalid Equation {self._prefix_list}'
        return stack.pop()

    def const_count(self):
        return self._const_count

    def func(self, X, constants=None):
        stack = deque()
        const_ind = 0
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.FLOAT_CONST or elem in rs_operators.INT_CONST:
                stack.append(float(elem))
                continue
            if elem in rs_operators.VARIABLES:
                stack.append(X[:,rs_operators.VARIABLES[elem]])
                continue
            if elem == rs_operators.CONST_SYMBOL:
                if constants is not None and const_ind < len(constants):
                    stack.append(constants[const_ind])
                    const_ind += 1
                else:
                    raise ConstantsCountError(f'not enough constants passed {self._prefix_list}, {constants}')
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    raise InvalidEquationError(f'Invalid Equation {self._prefix_list}')
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.func(*args))
                continue
            raise InvalidEquationError(f'Invalid symbol in Equation {self._prefix_list}')
        if len(stack) != 1:
            raise InvalidEquationError(f'Invalid Equation {self._prefix_list}')
        return stack.pop()

    def validate(self):
        self._const_count = 0
        complexity = 0
        stack = deque()
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.VARIABLES or elem == rs_operators.CONST_SYMBOL \
                        or elem in rs_operators.FLOAT_CONST or elem in rs_operators.INT_CONST:
                stack.append(elem)
                complexity += rs_operators.VAR_CONST_COMPLEXITY
                if elem == rs_operators.CONST_SYMBOL:
                    self._const_count += 1
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                complexity += operator.complexity
                if len(stack) < operator.arity:
                    return f'Invalid Equation {self._prefix_list}', None
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.repr(*args))
                continue
            return f'Invalid symbol in Equation {self._prefix_list}', None
        if len(stack) != 1:
            return f'Invalid Equation {self._prefix_list}', None
        self._repr = stack.pop()
        return 'OK', complexity

    def sympy_expr(self, constants=None):
        stack = deque()
        const_ind = 0
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.FLOAT_CONST:
                stack.append(float(elem))
                continue
            if elem in rs_operators.INT_CONST:
                stack.append(int(elem))
                continue
            if elem in rs_operators.VARIABLES:
                stack.append(sp.Symbol(elem))
                continue
            if elem == rs_operators.CONST_SYMBOL:
                if constants is not None and const_ind < len(constants):
                    stack.append(constants[const_ind])
                    const_ind += 1
                else:
                    raise ConstantsCountError(f'not enough constants passed {self._prefix_list}, {constants}')
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    return f'Invalid Equation {self._prefix_list}', None
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.sympy(*args))
                continue
            return f'Invalid symbol in Equation {self._prefix_list}', None
        if len(stack) != 1:
            return f'Invalid Equation {self._prefix_list}', None
        return stack.pop()

    @staticmethod
    def sympy_to_sting(sympy_expr):
        f_list = []
        stack = deque()
        stack.append(sympy_expr)
        while True:
            if len(stack) == 0:
                break
            root = stack.pop()
            f, a = root.func, root.args
            n_args = len(root.args)
            if isinstance(root, sp.core.numbers.Float):
                if str(root) in rs_operators.FLOAT_CONST:
                    f_list.append(str(root))
                else:
                    f_list.append(rs_operators.CONST_SYMBOL)
            elif isinstance(root, sp.core.numbers.Integer):
                if str(root) in rs_operators.INT_CONST:
                    f_list.append(str(root))
                elif str(-root) in rs_operators.INT_CONST and '-1' in rs_operators.INT_CONST:
                    f_list.extend(['mul', '-1', str(-root)])
                else:
                    f_list.append(rs_operators.CONST_SYMBOL)
            elif isinstance(root, sp.core.numbers.Half):
                if '0.5' in rs_operators.FLOAT_CONST:
                    f_list.append('0.5')
                else:
                    f_list.append(rs_operators.CONST_SYMBOL)
            elif isinstance(root, sp.core.power.Pow):
                f_list.extend(['pow'])
            elif isinstance(root, sp.core.add.Add):
                f_list.extend(['add'] * (n_args - 1))
            elif isinstance(root, sp.core.mul.Mul):
                f_list.extend(['mul'] * (n_args - 1))
            elif isinstance(root, sp.core.symbol.Symbol):
                f_list.append(root.name)
            elif isinstance(root, sp.sin):
                f_list.append('sin')
            elif isinstance(root, sp.cos):
                f_list.append('cos')
            elif isinstance(root, sp.log):
                f_list.append('log')
            else:
                f_list.append(f)
            stack.extend(a[::-1])
        return f_list
