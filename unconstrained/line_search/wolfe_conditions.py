import numpy as np
import matplotlib.pyplot as plt


class Wolfe:
    """
    This class gets a function, a point, and a descent direction, and analyzes the proper step lengths for the descent
    direction
    """

    __func = None
    __domain1 = None
    __domain2 = None
    __dimension = None
    __derivatives = list()
    __gradients = None
    __current_point = None
    __descent_direction = None
    __step_length = None
    __constant1 = None
    __constant2 = None
    __delta1 = None
    __delta2 = None

    def __init__(self, func, current_point, descent_direction, c1, c2=None):
        self.__func = func.get_function()
        self.__domain1 = func.get_domain1()
        self.__domain2 = func.get_domain2()
        self.__dimension = func.get_dimension()
        self.__current_point = current_point
        self.__descent_direction = descent_direction
        self.__constant1 = c1
        self.__constant2 = c2

        if self.__dimension == 1:
            self.__derivative()
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
        elif self.__dimension == 2:
            self.__gradient()
            self.__delta1 = self.__domain1[1] - self.__domain1[0]
            self.__delta2 = self.__domain2[1] - self.__domain2[0]

    def __derivative(self):
        """
        This method calculates the derivative of a 1D function at all the points of the __domain1
        :return: None, fills the __derivatives property with the derivative of __func at points of __domain1
        """
        for i in range(len(self.__domain1)):
            if i == len(self.__domain1) - 1:
                d = (self.__func(self.__domain1[i-1]) - self.__func(self.__domain1[i])) / \
                    (self.__domain1[i-1] - self.__domain1[i])
                self.__derivatives.append(d)
            else:
                d = (self.__func(self.__domain1[i + 1]) - self.__func(self.__domain1[i])) / \
                    (self.__domain1[i + 1] - self.__domain1[i])
                self.__derivatives.append(d)

    def __gradient(self):
        """
        This method calculates the gradient of the function through all points of the domain
        :return: None, fills the __gradient property with the gradient values of the function __func at all the points
        of the domain1 and domain2
        """
        x, y = np.meshgrid(self.__domain1, self.__domain2, sparse=False, indexing='ij')
        z = self.__func(x, y)

        self.__gradients = np.zeros((z.shape[0], z.shape[0], 2))

        for i in range(len(x)):
            for j in range(len(y)):
                if i == len(x) - 1:
                    self.__gradients[i, j, 0] = (z[i - 1, j] - z[i, j]) / (x[i - 1, j] - x[i, j])
                else:
                    self.__gradients[i, j, 0] = (z[i + 1, j] - z[i, j]) / (x[i + 1, j] - x[i, j])

                if j == len(y) - 1:
                    self.__gradients[i, j, 1] = (z[i, j - 1] - z[i, j]) / (y[i, j - 1] - y[i, j])
                else:
                    self.__gradients[i, j, 1] = (z[i, j + 1] - z[i, j]) / (y[i, j + 1] - y[i, j])

    def __calc_derivative(self, point):
        """
        This method calculates the derivative of a 1D function a given point
        :param point: denotes the point in which we aim to calculate the derivative
        :return: the derivative of the function '__func' at the given point
        """
        z1 = self.__func(point)
        z2 = self.__func(point + self.__delta1)

        return (z2 - z1) / self.__delta1

    def __calc_gradient(self, point_x, point_y):
        """
        This method calculates the gradient of a 2D function at a given point, decomposed by point_x and point_y
        :param point_x: denotes the x component of the point
        :param point_y: denotes the y component of the point
        :return: g : the gradient of the function '__func' at the given point
        """
        g = np.zeros((2,))

        z11 = self.__func(point_x, point_y)
        z12 = self.__func(point_x + self.__delta1, point_y)
        z21 = self.__func(point_x, point_y + self.__delta2)

        g1 = (z12 - z11) / self.__delta1
        g2 = (z21 - z11) / self.__delta2

        g[0] = g1
        g[1] = g2

        return g

    def get_acceptable_alpha(self, use_second_condition=False):
        """
        This method calculates the the acceptable interval for step length values
        :param use_second_condition: if this parameter is set to True, the method takes the second wolfe condition or
        the curvature condition to account.
        :return: proper_list, containing the valid values of alpha
        """
        proper_alphas = list()

        if self.__dimension == 1:
            upper_bound_alphas = set()
            lower_bound_alpha = set()

            # alpha_max = int(np.ceil((self.__domain1[-1] - self.__current_point) / self.__descent_direction))
            alpha_max = 50
            alpha_range = np.arange(0, alpha_max+0.01, 0.01)

            b = self.__func(self.__current_point)
            a = self.__constant1 * self.__calc_derivative(self.__current_point) * self.__descent_direction

            for alpha in alpha_range:
                f_value = self.__func(self.__current_point + alpha * self.__descent_direction)
                target_value = b + alpha * a

                if f_value <= target_value:
                    upper_bound_alphas.add(alpha)

            if use_second_condition:
                for alpha in alpha_range:
                    f_value = self.__calc_derivative(self.__current_point + alpha * self.__descent_direction)
                    curvature = self.__calc_derivative(self.__current_point) * self.__descent_direction

                    if f_value >= self.__constant2 * curvature:
                        lower_bound_alpha.add(alpha)

            if lower_bound_alpha:
                alpha_set = upper_bound_alphas.intersection(lower_bound_alpha)
            else:
                alpha_set = upper_bound_alphas

            for alpha in alpha_set:
                proper_alphas.append(alpha)

        elif self.__dimension == 2:
            upper_bound_alphas = set()
            lower_bound_alpha = set()

            # alpha_max1 = int(np.ceil((self.__domain1[-1] - self.__current_point[0]) / self.__descent_direction[0]))
            # alpha_max2 = int(np.ceil((self.__domain2[-1] - self.__current_point[1]) / self.__descent_direction[1]))
            # alpha_max = abs(max(alpha_max1, alpha_max2))

            alpha_max = 50

            alpha_range = np.arange(0, alpha_max+0.01, 0.01)

            b = self.__func(self.__current_point[0], self.__current_point[1])
            a = np.matmul(self.__calc_gradient(self.__current_point[0], self.__current_point[1]).reshape(1, 2),
                          self.__descent_direction.reshape(2, 1)) * self.__constant1
            a = a[0, 0]

            for alpha in alpha_range:
                f_point = self.__current_point + alpha * self.__descent_direction
                f_value = self.__func(f_point[0], f_point[1])

                if f_value <= b + alpha * a:
                    upper_bound_alphas.add(alpha)

            if use_second_condition:
                for alpha in alpha_range:
                    point = self.__current_point + alpha * self.__descent_direction
                    f_value = np.matmul(self.__calc_gradient(point[0], point[1]), self.__descent_direction)
                    curvature = np.matmul(self.__calc_gradient(self.__current_point[0], self.__current_point[1]),
                                          self.__descent_direction) * self.__constant2

                    if f_value >= curvature:
                        lower_bound_alpha.add(alpha)

            if lower_bound_alpha:
                alpha_set = upper_bound_alphas.intersection(lower_bound_alpha)
            else:
                alpha_set = upper_bound_alphas

            for alpha in alpha_set:
                proper_alphas.append(alpha)

        return proper_alphas

    def check_first_condition(self, point, alpha):
        if self.__dimension == 1:
            b = self.__func(point)
            a = self.__constant1 * self.__calc_derivative(point) * self.__descent_direction
            target_value = b + alpha * a
            f_value = self.__func(point + alpha * self.__descent_direction)
            if f_value <= target_value:
                return True
            return False
        else:
            b = self.__func(point[0], point[1])
            a = self.__constant1 * np.matmul(self.__calc_gradient(point[0], point[1]), self.__descent_direction)
            target_value = b + alpha * a
            f_point = point + alpha * self.__descent_direction
            f_value = self.__func(f_point[0], f_point[1])
            if f_value <= target_value:
                return True
            return False

    def draw_acceptable_alpha(self, use_second_condition=False):
        if self.__dimension == 1:
            alpha_max = int(np.ceil((self.__domain1[-1] - self.__current_point) / self.__descent_direction))

            alpha_points = np.arange(0, alpha_max, 0.1)
            phi = list()
            line = list()

            for i in range(len(alpha_points)):
                phi.append(self.__func(self.__current_point + alpha_points[i] * self.__descent_direction))
                line.append(self.__func(self.__current_point) + self.__constant1 * alpha_points[i] *
                            self.__calc_derivative(self.__current_point) * self.__descent_direction)

            if use_second_condition:
                phi_prime = list()
                phi_zero = list()

                for i in range(len(alpha_points)):
                    phi_prime.append(self.__calc_derivative(self.__current_point + alpha_points[i] *
                                                            self.__descent_direction) * self.__descent_direction)
                    phi_zero.append(self.__constant2 * self.__calc_derivative(self.__current_point) *
                                    self.__descent_direction)

                plt.subplot(2, 1, 1)
                plt.plot(alpha_points, phi, 'b')
                plt.plot(alpha_points, line, 'r')
                plt.title('Sufficient Decrease Condition')
                plt.xlabel('alpha')
                plt.legend(['f(x+alpha*p)', 'f(x)+c1*alpha*f\'(x)p'])
                plt.subplot(2, 1, 2)
                plt.plot(alpha_points, phi_prime, 'b')
                plt.plot(alpha_points, phi_zero, 'r')
                plt.title('Curvature Condition')
                plt.xlabel('alpha')
                plt.legend(['phi\'(alpha)', 'c2*phi\'(0)'])

                plt.tight_layout()
                plt.show()
            else:
                plt.plot(alpha_points, phi, 'b')
                plt.plot(alpha_points, line, 'r')
                plt.title('Sufficient Decrease Condition')
                plt.xlabel('alpha')
                plt.legend(['f(x+alpha*p)', 'f(x)+c1*alpha*f\'(x)p'])

                plt.tight_layout()
                plt.show()

        elif self.__dimension == 2:
            alpha_max1 = int(np.ceil((self.__domain1[-1] - self.__current_point[0]) / self.__descent_direction[0]))
            alpha_max2 = int(np.ceil((self.__domain2[-1] - self.__current_point[1]) / self.__descent_direction[1]))
            alpha_max = max(alpha_max1, alpha_max2)

            alpha_points = np.arange(0, alpha_max, 0.1)

            phi = list()
            line = list()

            m = np.matmul(self.__calc_gradient(self.__current_point[0], self.__current_point[1]),
                          self.__descent_direction)
            const = self.__func(self.__current_point[0], self.__current_point[1])

            for i in range(len(alpha_points)):
                f_point = self.__current_point + alpha_points[i] * self.__descent_direction
                phi.append(self.__func(f_point[0], f_point[1]))
                line.append(const + self.__constant1 * alpha_points[i] * m)

            if use_second_condition:
                phi_prime = list()
                phi_zero = list()

                for i in range(len(alpha_points)):
                    g_point = self.__current_point + alpha_points[i] * self.__descent_direction
                    phi_prime.append(np.matmul(self.__calc_gradient(g_point[0], g_point[1]), self.__descent_direction))
                    phi_zero.append(self.__constant2 * m)

                plt.subplot(2, 1, 1)
                plt.plot(alpha_points, phi, 'b')
                plt.plot(alpha_points, line, 'r')
                plt.title('Sufficient Decrease Condition')
                plt.xlabel('alpha')
                plt.legend(['f(x+alpha*p)', 'f(x)+c1*alpha*f\'(x)p'])
                plt.subplot(2, 1, 2)
                plt.plot(alpha_points, phi_prime, 'b')
                plt.plot(alpha_points, phi_zero, 'r')
                plt.title('Curvature Condition')
                plt.xlabel('alpha')
                plt.legend(['phi\'(alpha)', 'c2*phi\'(0)'])

                plt.tight_layout()
                plt.show()
            else:
                plt.plot(alpha_points, phi, 'b')
                plt.plot(alpha_points, line, 'r')
                plt.title('Sufficient Decrease Condition')
                plt.xlabel('alpha')
                plt.legend(['f(x+alpha*p)', 'f(x)+c1*alpha*f\'(x)p'])

                plt.tight_layout()
                plt.show()
