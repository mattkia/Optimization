import numpy as np


class RegionRadius:
    """
    This class implements the Trust Region algorithm to find the proper trust region radius in each iteration
    """

    __func = None
    __model = None
    __max_delta = None
    __eta = None
    __dimension = None
    __previous_delta = None
    __current_point = None
    __domain1 = None
    __domain2 = None
    __delta1 = None
    __delta2 = None
    __approximate_p = None

    def __init__(self, func, max_delta, eta, current_point, previous_delta=None):
        self.__func = func.get_function()
        self.__dimension = func.get_dimension()
        self.__max_delta = max_delta
        self.__eta = eta
        self.__current_point = current_point
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()

        if previous_delta is None:
            self.__previous_delta = np.random.rand(1,)[0] * max_delta
        else:
            self.__previous_delta = previous_delta

        if self.__dimension == 1:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        else:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
            self.__delta2 = self.__domain2[1] - self.__domain2[0]

        if self.__dimension == 1:
            self.__model = lambda p: self.__func(p) + self.__calc_derivative()*p + \
                                     0.5*(p**2)*self.__calc_second_derivative()
        else:
            self.__model = lambda p: self.__func(p[0], p[1]) + \
                                     np.matmul(self.__calc_gradient().reshape(1, 2), p.reshape(2, 1)) \
                                     + 0.5 * np.matmul(np.matmul(p.reshape(1, 2), self.__calc_hessian()),
                                                       p.reshape(2, 1))

    def __calc_derivative(self):
        z1 = self.__func(self.__current_point)
        z2 = self.__func(self.__current_point + self.__delta1)

        d = (z2 - z1) / self.__delta1

        return d

    def __calc_gradient(self):
        g = np.zeros((2,))

        z11 = self.__func(self.__current_point[0], self.__current_point[1])
        z12 = self.__func(self.__current_point[0] + self.__delta1, self.__current_point[1])
        z21 = self.__func(self.__current_point[0], self.__current_point[1] + self.__delta2)

        g[0] = (z12 - z11) / self.__delta1
        g[1] = (z21 - z11) / self.__delta2

        return g

    def __calc_second_derivative(self):
        z11 = self.__func(self.__current_point)
        z12 = self.__func(self.__current_point + self.__delta1)
        z21 = self.__func(self.__current_point - self.__delta1)

        d2 = (z12 + z21 - 2 * z11) / (self.__delta1 ** 2)

        return d2

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

    def __calc_approximate_p(self):
        if self.__dimension == 1:
            inverse_hessian = 1 / self.__calc_second_derivative()
            gradient = self.__calc_derivative()
            p = -inverse_hessian * gradient
        else:
            inverse_hessian = -np.linalg.inv(self.__calc_hessian())
            gradient = self.__calc_gradient()
            p = -np.matmul(inverse_hessian, gradient.reshape(2, 1))

        return p

    def __calc_ro(self):
        p = self.__calc_approximate_p()
        if self.__dimension == 1:
            nom = self.__func(self.__current_point) - self.__func(self.__current_point + p)
            denom = self.__model(0) - self.__model(p)
            return nom / denom
        else:
            nom = self.__func(self.__current_point[0], self.__current_point[1]) - \
                  self.__func(self.__current_point[0] + p[0], self.__current_point[1] + p[1])

            denom = self.__model(np.array([0, 0])) - self.__model(p)

            return nom / denom

    def calc_delta(self):
        ro = self.__calc_ro()

        if ro < 0.25:
            return self.__previous_delta / 4
        else:
            if ro > 0.75:
                return min(2 * self.__previous_delta, self.__max_delta)
            else:
                return self.__previous_delta

    def get_ro_list(self):
        return self.__ro_list
