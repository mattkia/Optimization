from constrained.problem import Problem
import numpy as np


class ConstrainedNewton:
    """
    This class implements the Newton method for solving the equality constrained convex problems. All the problems
    passed to this class are considered convex.
    :param: __problem : defines the equality constraint convex optimization problem
    :param: __initial_point : defines the initial point of the procedure. This point must be strictly feasible
    :param: __current_point : defines the current point the algorithm is working on
    :param: __KKT_matrix : defines the well-known KKT coefficient matrix for quadratic problems
    :param: __KKT_constant_vector : defines the well-known KKT constant vector for quadratic problems
    :param: __newton_step_param : defines the newton step at each level of the algorithm, and is the solution of
    the linear system imposed by the KKT coefficient matrix and KKT constant vector
    :param: __tolerance : defines a tolerance at which the algorithm terminates
    """
    __problem = None
    __initial_point = None
    __current_point = None
    __KKT_matrix = None
    __KKT_constant_vector = None
    __newton_step_param = None
    __newton_decrement_param = None
    __tolerance = None

    def __init__(self, problem, initial_point, tolerance):
        self.__problem = problem
        self.__initial_point = initial_point
        self.__current_point = initial_point
        self.__tolerance = tolerance

    def __newton_step(self):
        """
        This method calculates the newton step size at each point
        :return: None - resets the __newton_step_param variable of the class
        """
        objective = self.__problem.get_function()
        hessian = Problem.calc_hessian(objective, self.__current_point)
        gradient = Problem.calc_gradient(objective, self.__current_point)
        coefficient_matrix = self.__problem.get_coefficient_matrix()

        # Creating the KKT coefficient matrix from the objective function hessian and equality constraints
        # coefficient matrix. Creating the KKT constant vector from the objective function gradient.
        if coefficient_matrix is not None:
            self.__KKT_matrix = np.zeros((hessian.shape[0] + coefficient_matrix.shape[0],
                                          hessian.shape[0] + coefficient_matrix.shape[0]))
            self.__KKT_constant_vector = np.zeros((hessian.shape[0] + coefficient_matrix.shape[0], ))

            for i in range(hessian.shape[0]):
                for j in range(hessian.shape[0]):
                    self.__KKT_matrix[i, j] = hessian[i, j]

            for i in range(hessian.shape[0], hessian.shape[0] + coefficient_matrix.shape[0]):
                for j in range(hessian.shape[0]):
                    self.__KKT_matrix[i, j] = coefficient_matrix[i - hessian.shape[0], j]

            for i in range(hessian.shape[0]):
                for j in range(hessian.shape[0], hessian.shape[0], hessian.shape[0] + coefficient_matrix.shape[0]):
                    self.__KKT_matrix[i, j] = coefficient_matrix[j - hessian.shape[0], i]

            for i in range(gradient.shape[0]):
                self.__KKT_constant_vector[i] = -gradient[i]

        else:
            self.__KKT_matrix = hessian
            self.__KKT_constant_vector = -gradient

        # Calculating the solution of the KKT linear system
        solution = np.linalg.solve(self.__KKT_matrix, self.__KKT_constant_vector)

        newton_step = np.zeros(gradient.shape)

        for i in range(gradient.shape[0]):
            newton_step[i] = solution[i]

        # Updating the __newton_step_param
        self.__newton_step_param = newton_step

    def __newton_decrement(self):
        """
        This method calculates the newton decrement parameter
        :return: None - resets the __newton_decrement_param of the class
        """
        objective = self.__problem.get_function()
        hessian = Problem.calc_hessian(objective, self.__current_point)

        newton_decrement = np.matmul(np.matmul(self.__newton_step_param.T, hessian), self.__newton_step_param)

        self.__newton_decrement_param = newton_decrement

    def check_feasibility(self, test_point):
        inequalities = self.__problem.get_inequalities()
        feasible = True

        for i in range(len(inequalities)):
            if inequalities[i](test_point) >= 0:
                feasible = False
                break

        return feasible

    def run(self):
        while True:
            print('\t[*] Calculating the newton step')
            self.__newton_step()
            print('\t', self.__newton_step_param)
            print('\t[*] Calculating the newton decrement')
            self.__newton_decrement()
            print('\t', self.__newton_decrement_param)
            if self.__newton_decrement_param**2 / 2 <= self.__tolerance:
                break
            print('\t[*] Updating the current point')
            self.__current_point += self.__newton_step_param

        return self.__current_point

