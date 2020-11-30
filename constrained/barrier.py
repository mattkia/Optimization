import numpy as np
import matplotlib.pyplot as plt
from constrained.constrained_newton import ConstrainedNewton
from constrained.problem import Problem


class Barrier:
    """
    This class implements the Barrier algorithm for solving a general convex optimization problem
    :param: __initial_point : defines a strictly feasible starting point
    :param: __multiplicand : defines the 't' parameter of the interior point method
    :param: __mu : defines the increment coefficient of the the 't' parameter
    :param: __tolerance : defines the termination condition of the problem
    :param: __problem : defines the convex optimization problem; it's an instance of the Problem class
    """
    __initial_point = None
    __current_point = None
    __initial_t = None
    __current_t = None
    __mu = None
    __tolerance = None
    __problem = None
    __inequalities = None
    __coefficient_matrix = None
    __constant_vector = None
    __first_function = None
    __duality_gaps = []

    def __init__(self, problem, initial_point, multiplicand, mu, tolerance):
        self.__initial_point = initial_point
        self.__current_point = initial_point
        self.__initial_t = multiplicand
        self.__current_t = multiplicand
        self.__mu = mu
        self.__tolerance = tolerance
        self.__inequalities = problem.get_inequalities()
        self.__coefficient_matrix = problem.get_coefficient_matrix()
        self.__constant_vector = problem.get_constant_vector()
        self.__first_function = problem.get_function()

    def run(self):
        inequalities = self.__inequalities
        phi = lambda x: sum([-np.log(-inequalities[i](x)) for i in range(len(inequalities))])

        first_problem = self.__first_function
        iteration = 1
        print('[*] Initial Point : ', self.__initial_point)
        while True:
            print('[*] Iteration : ', iteration)
            iteration += 1
            # Step 0: Defining the new problem
            new_function = lambda x: self.__current_t * first_problem(x) + phi(x)
            new_problem = Problem(new_function, self.__inequalities, self.__coefficient_matrix, self.__constant_vector)
            self.__problem = new_problem
            print('[*] New problem is defined successfully')

            # Step 1: solving the new problem
            new_newton = ConstrainedNewton(self.__problem, self.__current_point, self.__tolerance)
            new_point = new_newton.run()

            # Step 2: updating the current point
            self.__current_point = new_point
            print('[*] Next Point : ', self.__current_point)

            # Step 3: Checking the stopping criterion
            m = len(inequalities)
            self.__duality_gaps.append(m/self.__current_t)
            if m/self.__current_t <= self.__tolerance:
                break

            # Step 4: Increment t
            self.__current_t = self.__current_t * self.__mu
            print('[*] Next t : ', self.__current_t)

        return self.__current_point

    def draw_duality_gaps(self):
        plt.plot(self.__duality_gaps)
        plt.xlabel('iteration')
        plt.ylabel('duality gap')
        plt.title('Barrier Method Duality Gap')
        plt.tight_layout()
        plt.show()
