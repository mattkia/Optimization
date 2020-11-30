import numpy as np
from line_search.backtracking import Backtracking
import matplotlib.pyplot as plt


class QuasiNewton:
    """
    This class implements the quasi-newton method for calculating the minimum point of a function
    """

    __func = None
    __domain1 = None
    __domain2 = None
    __dimension = None
    __initial_point = None
    __current_point = None
    __previous_point = None
    __previous_point_hessian = None
    __previous_point_inverse_hessian = None
    __function = None
    __delta1 = None
    __delta2 = None
    __accuracy = list()

    def __init__(self, func, initial_point):
        self.__function = func
        self.__func = func.get_function()
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()
        self.__dimension = func.get_dimension()
        self.__initial_point = initial_point
        self.__current_point = initial_point
        self.__previous_point = initial_point

        if self.__dimension == 1:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        elif self.__dimension == 2:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
            self.__delta2 = self.__domain2[1] - self.__domain2[0]

        self.__calc_initial_hessian()

    def __calc_derivative(self, point):
        z1 = self.__func(point)
        z2 = self.__func(point + self.__delta1)

        d = (z2 - z1) / self.__delta1

        return d

    def __calc_gradient(self, point_x, point_y):
        g = np.zeros((2,))
        z11 = self.__func(point_x, point_y)
        z12 = self.__func(point_x + self.__delta1, point_y)
        z21 = self.__func(point_x, point_y + self.__delta2)

        g[0] = (z12 - z11) / self.__delta1
        g[1] = (z21 - z11) / self.__delta2

        return g

    def __calc_sr1_hessian(self):
        s = self.__previous_point - self.__current_point
        if self.__dimension == 1:
            y = self.__calc_derivative(self.__previous_point) - self.__calc_derivative(self.__current_point)
            nom = y - self.__previous_point_hessian * s
            nominator = nom ** 2
            denominator = nom * s

            new_point_hessian = self.__previous_point_hessian + nominator / denominator
        else:
            y = self.__calc_gradient(self.__previous_point[0], self.__previous_point[1]) - self.__calc_gradient(
                self.__current_point[0], self.__current_point[1])
            nom = y.reshape(2, 1) - np.matmul(self.__previous_point_hessian, s.reshape(2, 1))
            nominator = np.matmul(nom, np.transpose(nom))
            denominator = np.matmul(np.transpose(nom), s.reshape(2, 1))

            new_point_hessian = self.__previous_point_hessian + nominator / denominator

        self.__previous_point_hessian = new_point_hessian

        return new_point_hessian

    def __calc_bfgs_hessian(self):
        s = self.__previous_point - self.__current_point
        if self.__dimension == 1:
            y = self.__calc_derivative(self.__previous_point) - self.__calc_derivative(self.__current_point)

            nominator1 = (self.__previous_point_hessian ** 2) * (s ** 2)
            denominator1 = self.__previous_point_hessian * s ** 2

            nominator2 = y ** 2
            denominator2 = y * s

            new_point_hessian = self.__previous_point_hessian - nominator1 / denominator1 + nominator2 / denominator2
        else:
            y = self.__calc_gradient(self.__previous_point[0], self.__previous_point[1]) - self.__calc_gradient(
                self.__current_point[0], self.__current_point[1])

            nom11 = np.matmul(self.__previous_point_hessian, s.reshape(2, 1))
            nom12 = np.matmul(s.reshape(1, 2), self.__previous_point_hessian)

            nominator1 = np.matmul(nom11, nom12)
            denom1 = np.matmul(s.reshape(1, 2), self.__previous_point_hessian)
            denominator1 = np.matmul(denom1, s.reshape(2, 1))

            nominator2 = np.matmul(y.reshape(2, 1), y.reshape(1, 2))
            denominator2 = np.matmul(y.reshape(1, 2), s.reshape(2, 1))

            new_point_hessian = self.__previous_point_hessian - nominator1 / denominator1 + nominator2 / denominator2

        self.__previous_point_hessian = new_point_hessian

        return new_point_hessian

    def __calc_bfgs_inverse(self):
        s = self.__previous_point - self.__current_point
        if self.__dimension == 1:
            y = self.__calc_derivative(self.__previous_point) - self.__calc_derivative(self.__current_point)
            ro = 1 / (y * s)
            constant1 = ro * y * s
            constant2 = ro * s ** 2
            new_point_inverse_hessian = (1 - constant1) ** 2 * self.__previous_point_inverse_hessian + constant2
        else:
            y = self.__calc_gradient(self.__previous_point[0], self.__previous_point[1]) - self.__calc_gradient(
                self.__current_point[0], self.__current_point[1])
            ro = 1 / np.matmul(y.reshape(1, 2), s.reshape(2, 1))
            constant1 = np.identity(2) - ro * np.matmul(s.reshape(2, 1), y.reshape(1, 2))
            constant2 = ro * np.matmul(s.reshape(2, 1), s.reshape(1, 2))

            new_point_inverse_hessian = np.matmul(constant1, self.__previous_point_inverse_hessian)
            new_point_inverse_hessian = np.matmul(new_point_inverse_hessian, constant1) + constant2

        self.__previous_point_inverse_hessian = new_point_inverse_hessian

        return new_point_inverse_hessian

    def __calc_sr1_inverse(self):
        s = self.__previous_point - self.__current_point
        if self.__dimension == 1:
            y = self.__calc_derivative(self.__previous_point) - self.__calc_derivative(self.__current_point)
            nom = s - y * self.__previous_point_inverse_hessian
            nominator = nom ** 2
            denominator = nom * y

            new_point_inverse_hessian = self.__previous_point_inverse_hessian + nominator / denominator
        else:
            y = self.__calc_gradient(self.__previous_point[0], self.__previous_point[1]) - self.__calc_gradient(
                self.__current_point[0], self.__current_point[1])
            nom = s.reshape(2, 1) - np.matmul(self.__previous_point_inverse_hessian, y.reshape(2, 1))
            nominator = np.matmul(nom, np.transpose(nom))
            denominator = np.matmul(np.transpose(nom), y.reshape(2, 1))
            new_point_inverse_hessian = self.__previous_point_inverse_hessian + nominator / denominator

        self.__previous_point_inverse_hessian = new_point_inverse_hessian

        return new_point_inverse_hessian

    def __calc_descent_direction(self):
        if self.__dimension == 1:
            return -self.__previous_point_inverse_hessian * self.__calc_derivative(self.__current_point)
        else:
            return -np.matmul(self.__previous_point_inverse_hessian,
                              self.__calc_gradient(self.__current_point[0],
                                                   self.__current_point[1]).reshape(2, 1)).reshape(2,)

    def __calc_initial_hessian(self):
        if self.__dimension == 1:
            self.__previous_point_hessian = self.__calc_derivative(self.__initial_point)
            self.__previous_point_inverse_hessian = 1 / self.__previous_point_hessian
        else:
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

            self.__previous_point_hessian = h
            self.__previous_point_inverse_hessian = np.linalg.inv(h)

    def __check_descent_direction(self, direction, current_point):
        if self.__dimension == 1:
            if self.__calc_derivative(current_point) * direction < 0:
                return True
            else:
                return False
        else:
            if np.matmul(self.__calc_gradient(current_point[0],
                                              current_point[1]).reshape(1, 2), direction.reshape(2, 1)) < 0:
                return True
            else:
                return False

    def run(self, max_iter=100, mode='sr1'):
        iteration = 1
        while iteration <= max_iter:
            direction = -self.__calc_descent_direction()

            if self.__check_descent_direction(direction, self.__current_point) is False:
                print('Non-Descent Direction : ', direction)

            if self.__check_descent_direction(direction, self.__current_point) is False:
                if self.__dimension == 1:
                    d = self.__calc_derivative(self.__current_point)
                    direction = - d/abs(d)
                else:
                    g = self.__calc_gradient(self.__current_point[0], self.__current_point[1])
                    direction = - g / np.sqrt(g[0]**2 + g[1]**2)

            bt = Backtracking(self.__function, direction, self.__current_point, 0.1, 0.15)

            alpha = bt.get_alpha()

            if alpha is None:
                break

            new_point = self.__current_point + alpha * direction

            if self.__dimension == 1:
                acc = abs(self.__func(new_point) - self.__func(self.__current_point))
            else:
                acc = abs(self.__func(new_point[0], new_point[1]) -
                          self.__func(self.__current_point[0], self.__current_point[1]))

            self.__accuracy.append(acc)

            self.__previous_point = self.__current_point
            self.__current_point = new_point

            if mode == 'sr1':
                self.__calc_sr1_inverse()
            elif mode == 'bfgs':
                self.__calc_bfgs_inverse()

            # print('Iteration', iteration)
            # print('Descent direction : ', direction)
            # print('Selected alpha :', alpha)
            # print('New point : ', new_point)

            iteration += 1

        return self.__current_point

    def draw_accuracy_graph(self):
        plt.plot(self.__accuracy[:20])
        plt.title('|f(x_k+1) - f(x_k)|')
        plt.show()
