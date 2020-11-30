import sys
import numpy as np


if __name__ == '__main__':
    sys.path.append('../..')
    from optimization.utilities.function import Function


# 2D function test
dom1 = np.linspace(-2, 2, 51)
dom2 = np.linspace(-2, 2, 51)

a = Function(func=lambda x1, x2: np.sin(np.sqrt(x1**2 + x2**2)), domain1=dom1, domain2=dom2)
a.draw_function()

# 1D function test
c = Function(func=lambda x: x**3 + 2*x + 1, domain1=dom1)
c.draw_function()
