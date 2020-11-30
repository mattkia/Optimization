import numpy as np
from line_search.backtracking import Backtracking
import matplotlib.pyplot as plt


class SteepestDescent:
    """
    This class implements the Steepest Descent algorithm. The class instance is constructed with a function of Function
    type and an initial point
    @:param __func : is the lambda expression of the target function extracted from 'func'
    @:param __domain1 : is the function domain of the first dimension and is extracted from 'func'
    @:param __domain2 : is the function domain of the second dimension and is extracted from 'func'
    @:param __initial_point : is the initial point of the algorithm
    @:param __derivatives : is the array of the derivatives of the function in the domain1 points
    @:param __gradients : is the array of the gradients of the matrix in the (domain1, domain2) points
    """

    __func = None
    __domain1 = None
    __domain2 = None
    __dimension = None
    __initial_point = None
    __current_point = None
    __delta1 = None
    __delta2 = None
    __function = None
    __accuracy = list()

    def __init__(self, func, initial_point):
        self.__function = func
        self.__func = func.get_function()
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()
        self.__dimension = func.get_dimension()
        self.__initial_point = initial_point
        self.__current_point = initial_point

        if self.__dimension == 1:
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        elif self.__dimension == 2:
            self.__delta2 = self.__domain2[1] - self.__domain2[0]
            self.__delta1 = self.__domain1[1] - self.__domain1[0]

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

        g1 = (z12 - z11) / self.__delta1
        g2 = (z21 - z11) / self.__delta2

        g[0] = g1
        g[1] = g2

        return g

    def descent_direction(self, point):
        if self.__dimension == 1:
            direction = self.__calc_derivative(point)
            norm = abs(direction)
            return direction / norm
        elif self.__dimension == 2:
            direction = self.__calc_gradient(point[0], point[1])
            norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
            return direction / norm

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

    def run(self, max_iter=100):
        iteration = 1
        while iteration <= max_iter:
            direction = -self.descent_direction(self.__current_point)

            if self.__check_descent_direction(direction, self.__current_point) is False:
                print('Non-Descent Direction : ', direction)

            bt = Backtracking(self.__function, direction, self.__current_point, 0.1, 0.4)

            alpha = bt.get_alpha()

            if alpha is None:
                break

            new_point = self.__current_point + alpha * direction

            # print('Iteration', iteration)
            # print('Descent direction : ', direction)
            # print('Selected alpha :', alpha)
            # print('New point : ', new_point)

            if self.__dimension == 1:
                acc = abs(self.__func(new_point) - self.__func(self.__current_point))
            else:
                acc = abs(self.__func(new_point[0], new_point[1]) -
                          self.__func(self.__current_point[0], self.__current_point[1]))
            self.__accuracy.append(acc)

            self.__current_point = new_point

            iteration += 1

        return self.__current_point

    def draw_accuracy_graph(self):
        plt.plot(self.__accuracy[:20])
        plt.title('|f(x_k+1) - f(x_k)|')
        plt.show()

