import numpy as np
import sys

if __name__ == '__main__':
    sys.path.append('../..')
    from optimization.line_search.steepest_descent import SteepestDescent
    from optimization.line_search.wolfe_conditions import Wolfe
    from optimization.utilities.function import Function


dom1 = np.arange(-2, 2, 0.1)
dom2 = np.arange(-2, 2, 0.1)

a = Function(func=lambda x: x**2, domain1=dom1)
b = Function(func=lambda x1, x2: x1**2 + x2**2, domain1=dom1, domain2=dom2)

w1 = Wolfe(a, -1, 1.9, 0.5, c2=0.5)
# print(w1.get_acceptable_alpha())
# print(w1.get_acceptable_alpha(use_second_condition=True))
# w1.draw_acceptable_alpha()
# w1.draw_acceptable_alpha(use_second_condition=True)

w2 = Wolfe(b, np.array([-1, -1]), np.array([0.01, 0.01]), 0.5, 0.5)
# print(w2.get_acceptable_alpha())
# print(w2.get_acceptable_alpha(use_second_condition=True))
# w2.draw_acceptable_alpha()
# w2.draw_acceptable_alpha(use_second_condition=True)

sd1 = SteepestDescent(b, np.array([-1, -1]))
print(sd1.run())
