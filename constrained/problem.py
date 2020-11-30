import numpy as np


class Problem:
    """
    This class implements the structure of a convex optimisation problem.
    :param: __func : defines the objective function. It's expressed as a lambda expression
    :param: __inequality_constraints : defines a list of the inequality constraints as a list of lambda expressions
    :param: __coefficient_matrix : defines the coefficient matrix of the affine constraints
    :param: __constant_vector : defines the constant vector of the affine constraints
    """
    __func = None
    __inequality_constraints = list()
    __coefficients_matrix = None
    __constant_vector = None

    def __init__(self, func, inequality_constraints, coefficient_matrix, constant_vector):
        self.__func = func
        self.__inequality_constraints = inequality_constraints
        self.__coefficients_matrix = coefficient_matrix
        self.__constant_vector = constant_vector

    def get_function(self):
        return self.__func

    def get_coefficient_matrix(self):
        return self.__coefficients_matrix

    def get_constant_vector(self):
        return self.__constant_vector

    def get_inequalities(self):
        return self.__inequality_constraints

    @staticmethod
    def calc_gradient(func, point):
        """
        Method to calculate the gradient of a given function
        :param func : represents a function given in lambda expression form
        :param point : represents the point in which we want to calculate the gradient; remember that the point must
        always be given by a numpy array with shape (n,)
        :return gradient : returns the gradient of the function at the given point in the form of a numpy array with
        shape (n,)
        """

        gradient = np.zeros(point.shape)
        dimension = point.shape[0]
        delta = 0.001

        for i in range(dimension):
            additive = np.zeros(point.shape)
            additive[i] += delta
            right_point = point + additive
            left_point = point - additive
            derivative = (func(right_point) - func(left_point)) / (2 * delta)
            gradient[i] = derivative

        return gradient

    @staticmethod
    def calc_hessian(func, point):
        """
        This method calculates the hessian of a given function at a given point
        :param func: represents a given function given in lambda expression form
        :param point: represents the point in which we want to calculate the gradient; remember that the point must
        always be given by a numpy array with shape (n,)
        :return hessian: return the hessian of the function at the given point in the form of a numpy array with
        shape (n,n)
        """

        dimension = point.shape[0]
        hessian = np.zeros((dimension, dimension))
        delta = 0.001

        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    additive = np.zeros(point.shape)
                    additive[i] = delta
                    right_point = point + additive
                    left_point = point - additive
                    derivative = (func(right_point) - 2 * func(point) + func(left_point)) / delta ** 2
                    hessian[i, i] = derivative
                else:
                    additive1 = np.zeros(point.shape)
                    additive2 = np.zeros(point.shape)
                    additive1[i] = delta
                    additive1[j] = delta
                    additive2[i] = delta
                    additive2[j] = -delta
                    p1 = point + additive1
                    p2 = point - additive2
                    p3 = point + additive2
                    p4 = point - additive1
                    derivative = (func(p1) - func(p2) - func(p3) + func(p4)) / (4 * delta**2)
                    hessian [i, j] = derivative

        return hessian
