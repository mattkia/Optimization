import numpy as np
from line_search.wolfe_conditions import Wolfe


class Interpolation:
    """
    This class uses the cubic interpolation algorithm to find a proper step length
    """

    __func = None
    __function = None
    __dimension = None
    __constant1 = None
    __constant2 = None
    __current_point = None
    __descent_direction = None
    __delta1 = 0.01
    __delta2 = 0.01

    def __init__(self, func,current_point, descent_direction, c1, c2):
        self.__func = func.get_function()
        self.__function = func
        self.__dimension = func.get_dimension()
        self.__constant1 = c1
        self.__constant2 = c2
        self.__current_point = current_point
        self.__descent_direction = descent_direction

    def __calc_derivative(self):
        z1 = self.__func(self.__current_point - self.__delta1)
        z2 = self.__func(self.__current_point + self.__delta1)

        d = (z2 - z1) / (2 * self.__delta1)

        return d

    def __calc_second_derivative(self):
        z11 = self.__func(self.__current_point)
        z12 = self.__func(self.__current_point - self.__delta1)
        z21 = self.__func(self.__current_point + self.__delta2)

        d2 = (z12 - 2 * z11 + z21) / (self.__delta1 ** 2)

        return d2

    def __calc_gradient(self):
        g = np.zeros((2,))

        g11 = self.__func(self.__current_point[0] - self.__delta1, self.__current_point[1])
        g12 = self.__func(self.__current_point[0] + self.__delta1, self.__current_point[1])
        g21 = self.__func(self.__current_point[0], self.__current_point[1] - self.__delta2)
        g22 = self.__func(self.__current_point[0], self.__current_point[1] + self.__delta2)

        g[0] = (g12 - g11) / (2 * self.__delta1)
        g[1] = (g22 - g21) / (2 * self.__delta2)

        return g

    def __calc_hessian(self):
        h = np.zeros((2, 2))

        p00 = self.__func(self.__current_point[0], self.__current_point[1])
        p01 = self.__func(self.__current_point[0] + self.__delta1, self.__current_point[1])
        p02 = self.__func(self.__current_point[0] - self.__delta1, self.__current_point[1])
        h0 = (p01 + p02 - 2 * p00) / (self.__delta1 ** 2)

        p10 = self.__func(self.__current_point[0] + self.__delta1, self.__current_point[1] + self.__delta2)
        p11 = self.__func(self.__current_point[0], self.__current_point[1] + self.__delta2)
        p12 = self.__func(self.__current_point[0] + self.__delta1, self.__current_point[1])
        p13 = self.__func(self.__current_point[0], self.__current_point[1])
        h1 = (p10 - p11 - p12 + p13) / (self.__delta1 * self.__delta2)

        p30 = self.__func(self.__current_point[0], self.__current_point[1])
        p31 = self.__func(self.__current_point[0], self.__current_point[1] + self.__delta2)
        p32 = self.__func(self.__current_point[0], self.__current_point[1] - self.__delta2)
        h3 = (p31 + p32 - 2 * p30) / (self.__delta2 ** 2)

        h[0, 0] = h0
        h[0, 1] = h1
        h[1, 0] = h1
        h[1, 1] = h3

        return h

    def get_alpha(self):
        wolfe = Wolfe(self.__function, self.__current_point, self.__descent_direction,
                      self.__constant1, self.__constant2)
        step_lengths = wolfe.get_acceptable_alpha()

        if len(step_lengths) == 0:
            return None
        alpha0 = max(step_lengths)

        condition = wolfe.check_first_condition(self.__current_point, alpha0)

        return alpha0
