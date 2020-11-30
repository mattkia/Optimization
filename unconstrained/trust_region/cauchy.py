import numpy as np
import matplotlib.pyplot as plt
from trust_region.region_radius import RegionRadius


class CauchyPoint:
    """
    This class implements the cauchy point method. It finds the cauchy point in a trust region and find the next point
    in each iteration
    """

    __func = None
    __function = None
    __current_point = None
    __domain1 = None
    __domain2 = None
    __delta1 = None
    __delta2 = None
    __dimension = None
    __delta = None

    def __init__(self, func, current_point, delta):
        self.__func = func.get_function()
        self.__current_point = current_point
        self.__function = func
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()
        self.__dimension = func.get_dimension()
        self.__delta = delta

        if self.__dimension == 1:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        else:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
            self.__delta2 = self.__domain2[1] - self.__domain2[0]

    def __calc_derivative(self):
        z1 = self.__func(self.__current_point + self.__delta1)
        z2 = self.__func(self.__current_point)

        d = (z1 - z2) / self.__delta1

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
        z0 = self.__func(self.__current_point)
        z1 = self.__func(self.__current_point + self.__delta1)
        z2 = self.__func(self.__current_point - self.__delta1)

        d2 = (z1 + z2 - 2 * z0) / (self.__delta1 ** 2)

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

    @staticmethod
    def __calc_l2_norm(point_x, point_y):
        norm = np.sqrt(point_y ** 2 + point_x ** 2)
        return norm

    def calc_cauchy_point(self):
        if self.__dimension == 1:
            g = self.__calc_derivative()
            h = self.__calc_second_derivative()
            a = (g ** 2) * h
            if a > 0:
                taw1 = (abs(g) ** 3) / (a * self.__delta)
                taw = min(taw1, 1)
            else:
                taw = 1
        else:
            g = self.__calc_gradient()
            g_norm = self.__calc_l2_norm(g[0], g[1])
            h = self.__calc_hessian()
            const1 = np.matmul(g.reshape(1, 2), h)
            const2 = np.matmul(const1, g.reshape(2, 1))

            taw1 = (g_norm ** 3) / (self.__delta * const2)
            taw1 = taw1[0, 0]

            taw = min(taw1, 1)

        if self.__dimension == 1:
            d = self.__calc_derivative()
            constant = -taw * self.__delta / abs(d)
            return constant * d
        else:
            g = self.__calc_gradient()
            g_norm = self.__calc_l2_norm(g[0], g[1])
            constant = -taw * self.__delta / g_norm
            return constant * g


class CauchyTrustRegion:
    """
    This class uses the cauchy point class and runs iteratively to find the optimum point
    """

    __func = None
    __function = None
    __initial_point = None
    __delta = None
    __cauchy_point = None
    __current_point = None
    __dimension = None
    __delta1 = 0.01
    __delta2 = 0.01
    __accuracy = list()
    __ro_list = list()

    def __init__(self, func, initial_point):
        self.__func = func.get_function()
        self.__function = func
        self.__initial_point = initial_point
        self.__current_point = initial_point
        self.__dimension = func.get_dimension()

    def __calc_derivative(self, point):
        z1 = self.__func(point - self.__delta1)
        z2 = self.__func(point + self.__delta1)

        d = (z2 - z1) / (2 * self.__delta1)

        return d

    def __calc_second_derivative(self, point):
        z11 = self.__func(point)
        z12 = self.__func(point - self.__delta1)
        z21 = self.__func(point + self.__delta2)

        d2 = (z21 + z12 - 2 * z11) / (self.__delta1 ** 2)

        return d2

    def __calc_gradient(self, point_x, point_y):
        g = np.zeros((2,))

        z11 = self.__func(point_x - self.__delta1, point_y)
        z12 = self.__func(point_x + self.__delta1, point_y)
        z21 = self.__func(point_x, point_y - self.__delta2)
        z22 = self.__func(point_x, point_y + self.__delta2)

        g[0] = (z12 - z11) / (2 * self.__delta1)
        g[1] = (z22 - z21) / (2 * self.__delta2)

        return g

    def __calc_hessian(self, point_x, point_y):
        h = np.zeros((2, 2))

        p00 = self.__func(point_x, point_y)
        p01 = self.__func(point_x + self.__delta1, point_y)
        p02 = self.__func(point_x - self.__delta1, point_y)
        h0 = (p01 + p02 - 2 * p00) / (self.__delta1 ** 2)

        p10 = self.__func(point_x + self.__delta1, point_y + self.__delta2)
        p11 = self.__func(point_x, point_y + self.__delta2)
        p12 = self.__func(point_x + self.__delta1, point_y)
        p13 = self.__func(point_x, point_y)
        h1 = (p10 - p11 - p12 + p13) / (self.__delta1 * self.__delta2)

        p30 = self.__func(point_x, point_y)
        p31 = self.__func(point_x, point_y + self.__delta2)
        p32 = self.__func(point_x, point_y - self.__delta2)
        h3 = (p31 + p32 - 2 * p30) / (self.__delta2 ** 2)

        h[0, 0] = h0
        h[0, 1] = h1
        h[1, 0] = h1
        h[1, 1] = h3

        return h

    def run(self, max_iter=1000, threshold=0.01):
        initial_delta = RegionRadius(self.__function, 1, 0.1, self.__current_point)
        self.__delta = initial_delta.calc_delta()

        for iteration in range(max_iter):
            cauchy_point = CauchyPoint(self.__function, self.__current_point, self.__delta)

            pc = cauchy_point.calc_cauchy_point()

            if self.__dimension == 1:
                nominator = self.__func(self.__current_point) - self.__func(self.__current_point + pc)
                model = lambda p: self.__func(self.__current_point) + self.__calc_derivative(self.__current_point) * \
                                  p + 0.5 * p**2 * self.__calc_second_derivative(self.__current_point)
                denominator = model(0) - model(pc)
                self.__ro_list.append(nominator/denominator)
            else:
                nominator = self.__func(self.__current_point[0], self.__current_point[1]) - self.__func(self.__current_point[0] + pc[0], self.__current_point[1] + pc[1])
                model = lambda p: self.__func(self.__current_point[0], self.__current_point[1]) + np.matmul(self.__calc_gradient(self.__current_point[0], self.__current_point[1]).reshape(1, 2), p.reshape(2, 1)) + 0.5 * np.matmul(np.matmul(p.reshape(1, 2), self.__calc_hessian(self.__current_point[0], self.__current_point[1])), p.reshape(2, 1))
                denominator = model(np.array([0, 0])) - model(pc)

                self.__ro_list.append((nominator / denominator)[0, 0])

            if self.__dimension == 1:
                thresh = self.__func(pc) - self.__func(self.__current_point)
                self.__accuracy.append(abs(thresh))
            else:
                thresh = self.__func(pc[0], pc[1]) - self.__func(self.__current_point[0], self.__current_point[1])
                self.__accuracy.append(abs(thresh))

            self.__current_point = pc

            new_delta = RegionRadius(self.__function, 1, 0.1, self.__current_point, self.__delta)
            self.__delta = new_delta.calc_delta()

            if thresh <= threshold:
                break

        return self.__current_point

    def draw_accuracy_graph(self):
        plt.plot(self.__accuracy)
        plt.title('|f(x_k+1) - f(x_k)|')
        plt.show()

    def draw_ro_graph(self):
        plt.plot(self.__ro_list)
        plt.title('ro')
        plt.show()
