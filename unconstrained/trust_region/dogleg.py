import numpy as np
import matplotlib.pyplot as plt
from trust_region.region_radius import RegionRadius


class DogLeg:
    """
    This class implements dogleg method to find the optimum direction ina trust region at each iteration
    """

    __func = None
    __function = None
    __initial_point = None
    __delta = None
    __domain1 = None
    __domain2 = None
    __delta1 = None
    __delta2 = None
    __dimension = None
    __current_point = None
    __accuracy = list()
    __ro_list = list()

    def __init__(self, func, initial_point):
        self.__func = func.get_function()
        self.__initial_point = initial_point
        self.__function = func
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()
        self.__dimension = func.get_dimension()
        self.__current_point = initial_point

        if self.__dimension == 1:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        else:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
            self.__delta2 = self.__domain2[1] - self.__domain2[0]

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

    @staticmethod
    def __calc_inverse_hessian(hessian_matrix):
        inverse_hessian = np.linalg.inv(hessian_matrix)
        return inverse_hessian

    @staticmethod
    def __calc_l2_norm(point_x, point_y):
        norm = np.sqrt(point_x ** 2 + point_y ** 2)
        return norm

    def calc_dogleg_curve(self):
        if self.__dimension == 1:
            second_derivative = self.__calc_second_derivative()
            derivative = self.__calc_derivative()

            pb = -second_derivative * derivative

            pu = -((derivative ** 2) / (derivative ** 2 * second_derivative)) * derivative

            pu_norm = abs(pu)
        else:
            hessian_matrix = self.__calc_hessian()
            inverse_hessian = self.__calc_inverse_hessian(hessian_matrix)
            gradient = self.__calc_gradient()

            pb = -np.matmul(np.transpose(inverse_hessian), gradient.reshape(2, 1))

            pu = -(np.matmul(gradient.reshape(1, 2), gradient.reshape(2, 1)) /
                   np.matmul(np.matmul(gradient.reshape(1, 2), hessian_matrix), gradient.reshape(2, 1))) * gradient
            pu = pu.reshape(2,)
            pu_norm = self.__calc_l2_norm(pu[0], pu[1])

        if pu_norm >= self.__delta:
            taw = self.__delta / pu_norm
            return taw * pu
        else:
            if self.__dimension == 1:
                taw1 = (self.__delta - pu) / (pb - pu) + 1
                taw2 = (self.__delta - pu) / (pu - pb) + 1
                if 1 <= taw1 <= 2:
                    taw = taw1
                elif 1 <= taw2 <= 2:
                    taw = taw2
                else:
                    return pb
                return pu + (taw - 1) * (pb - pu)
            else:
                a = self.__calc_l2_norm(pb[0], pb[1]) ** 2 + self.__calc_l2_norm(pu[0], pu[1]) ** 2 - 2 * pu[1] * pb[1]
                b = -2 * self.__calc_l2_norm(pb[0], pb[1]) ** 2 + 2 * pu[1] * pb[1]
                c = self.__calc_l2_norm(pb[0], pb[1])

                taw1 = -b + np.sqrt(b ** 2 - 4 * a * c)
                taw2 = -b - np.sqrt(b ** 2 - 4 * a * c)

                if 1 <= taw1 <= 2:
                    taw = taw1
                elif 1 <= taw2 <= 2:
                    taw = taw2
                else:
                    return pb

                return (pu.reshape(2, 1) + (taw - 1) * (pb.reshape(2, 1) - pu.reshape(2, 1))).reshape(2,)

    def run(self, max_iter=1000, threshold=0.01):
        initial_delta = RegionRadius(self.__function, 1, 0.1, self.__current_point)
        self.__delta = initial_delta.calc_delta()

        for i in range(max_iter):
            new_point = self.calc_dogleg_curve()

            if self.__dimension == 1:
                nominator = self.__func(self.__current_point) - self.__func(self.__current_point + new_point)
                model = lambda p: self.__func(self.__current_point) + self.__calc_derivative() * p + 0.5 * p**2 * self.__calc_second_derivative()
                denominator = model(0) - model(new_point)

                self.__ro_list.append(nominator/denominator)
            else:
                nominator = self.__func(self.__current_point[0], self.__current_point[1]) - self.__func(self.__current_point[0] + new_point[0], self.__current_point[1] + new_point[1])
                model = lambda p: self.__func(self.__current_point[0], self.__current_point[1]) + np.matmul(self.__calc_gradient().reshape(1, 2), p.reshape(2, 1)) + 0.5 * np.matmul(np.matmul(p.reshape(1, 2), self.__calc_hessian()), p.reshape(2, 1))
                denominator = model(np.array([0, 0])) - model(new_point)

                self.__ro_list.append(nominator / denominator)

            if self.__dimension == 1:
                thresh = abs(self.__func(new_point) - self.__func(self.__current_point))
                self.__accuracy.append(thresh)
            else:
                thresh = abs(self.__func(new_point[0], new_point[1]) -
                             self.__func(self.__current_point[0], self.__current_point[1]))
                self.__accuracy.append(thresh)

            if thresh <= threshold:
                self.__current_point = new_point
                break

            self.__current_point = new_point

            new_delta = RegionRadius(self.__function, 1, 0.1, self.__current_point, self.__delta)
            self.__delta = new_delta.calc_delta()

        return self.__current_point

    def draw_accuracy_graph(self):
        plt.plot(self.__accuracy)
        plt.title('|f(x_k+1) - f(x_k)|')
        plt.show()

    def draw_ro_graph(self):
        plt.plot(self.__ro_list)
        plt.title('ro')
        plt.show()
