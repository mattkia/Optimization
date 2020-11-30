import matplotlib.pyplot as plt
import numpy as np
from constrained.problem import Problem


class PrimalDualInteriorPoint:
    """
    This class implements the Primal-Dual Interior Point method for a general constrained convex optimization problem.
    :param: __problem : defines the optimization problem
    :param: __current_point : defines the current primal point
    :param: __feasibility_tolerance : defines the convergence criteria for primal and dual norm
    :param: __eta_tolerance : defines the convergence criteria for surrogate duality gap
    :param: __mu : defines the constant multiplicand to increase the 't' parameter
    :param: __current_lambda : defines the dual vector for inequality constraints
    :param: __inequality_constraints : defines the list of inequality constraints
    :param: __objective_function : defines the convex objective function, which is extracted from the problem instance
    :param: __coefficient_matrix : defines the coefficient matrix of the affine equality constraint
    :param: __constant_vector : defines the constant vector of the equality constraint
    :param: __current_r_dual : defines the dual part of the primal-dual vector
    :param: __current_r_primal : defines the primal part of the primal-dual vector
    :param: __current_r_centrality : defines the centrality part of the primal-dual vector
    :param: __surrogate_duality_gaps : defines the list of the duality gaps; used to draw the duality gap graph
    """
    __problem = None
    __current_point = None
    __primal_step = None
    __feasibility_tolerance = None
    __eta_tolerance = None
    __mu = None
    __current_lambda = None
    __inequality_constraints = None
    __objective_function = None
    __coefficient_matrix = None
    __constant_vector = None
    __current_r_dual = None
    __current_r_centrality = None
    __surrogate_duality_gaps = []
    __plots = []
    __dual_norms = []

    def __init__(self, problem, initial_point, initial_lambda, feasibility_tolerance, eta_tolerance, mu):
        self.__problem = problem
        self.__current_point = initial_point
        self.__current_lambda = initial_lambda
        self.__feasibility_tolerance = feasibility_tolerance
        self.__eta_tolerance = eta_tolerance
        self.__mu = mu
        self.__inequality_constraints = problem.get_inequalities()
        self.__objective_function = problem.get_function()
        self.__coefficient_matrix = problem.get_coefficient_matrix()
        self.__constant_vector = problem.get_constant_vector()

    def make_dual_matrix(self):
        """
        This method calculates the coefficient matrix of the primal-dual problem
        :return: dual_matrix
        """
        # creating the differential matrix of the inequalities
        m = len(self.__inequality_constraints)
        index = 0
        differential_matrix = np.zeros((m, 50))
        for inequality in self.__inequality_constraints:
            gradient = Problem.calc_gradient(inequality, self.__current_point).reshape(1, 50)
            differential_matrix[index] = gradient
            index += 1

        # creating the objective function hessian
        objective_hessian = Problem.calc_hessian(self.__objective_function, self.__current_point)

        # creating the hessian matrix of the inequality constraints
        inequality_hessians = []
        for i in range(len(self.__inequality_constraints)):
            hessian = Problem.calc_hessian(self.__inequality_constraints[i], self.__current_point)
            inequality_hessians.append(self.__current_lambda[i]*hessian)

        # creating the lambda diagonal matrix
        diagonal_lambda = np.identity(m)
        for i in range(m):
            diagonal_lambda[i, i] = self.__current_lambda[i]

        # creating the diagonal inequalities
        diagonal_inequalities = np.identity(m)
        for i in range(m):
            diagonal_inequalities[i, i] = self.__inequality_constraints[i](self.__current_point)

        # creating the dual coefficient matrix
        dual_matrix = np.zeros((50+m, 50+m))
        dual_matrix[0:50, 0:50] = objective_hessian + sum(inequality_hessians)
        dual_matrix[0:50, 50:50+m] = differential_matrix.transpose()
        dual_matrix[50:50+m, 0:50] = -np.matmul(diagonal_lambda, differential_matrix)
        dual_matrix[50:50+m, 50:50+m] = -diagonal_inequalities

        return dual_matrix

    def make_dual_constant_vector(self):
        """
        This method calculates the constant vector in the primal-dual problem
        :return: constant-vector
        """
        m = len(self.__inequality_constraints)

        index = 0
        differential_matrix = np.zeros((m, 50))
        for inequality in self.__inequality_constraints:
            gradient = Problem.calc_gradient(inequality, self.__current_point).reshape(1, 50)
            differential_matrix[index] = gradient
            index += 1

        diagonal_lambda = np.identity(m)
        for i in range(m):
            diagonal_lambda[i, i] = self.__current_lambda[i]

        inequality_matrix = np.zeros((m, 1))
        for i in range(m):
            inequality_matrix[i] = self.__inequality_constraints[i](self.__current_point)

        eta = -np.matmul(inequality_matrix.reshape(1, m), self.__current_lambda.reshape(m, 1))
        self.__surrogate_duality_gaps.append(eta[0][0])
        current_t = self.__mu * m / eta

        constant_vector = np.zeros((50+m, 1))

        r_dual = Problem.calc_gradient(self.__objective_function, self.__current_point).reshape(50, 1) + \
                 np.matmul(differential_matrix.transpose(), self.__current_lambda)
        r_centrality = -np.matmul(diagonal_lambda, inequality_matrix) - (1/current_t)*np.ones((m, 1))

        self.__current_r_dual = r_dual
        self.__current_r_centrality = r_centrality

        constant_vector[0:50] = -r_dual
        constant_vector[50:50+m] = -r_centrality

        return constant_vector

    def check_primal_feasibility(self):
        m = len(self.__inequality_constraints)

        feasible = True

        for i in range(m):
            if self.__inequality_constraints[i](self.__current_point) >= 0:
                feasible = False
                break

        return feasible

    def check_dual_feasibility(self):
        m = self.__current_lambda.shape[0]

        feasible = True

        for i in range(m):
            if self.__current_lambda[i] <= 0:
                feasible = False
                break

        return feasible

    def run(self):
        m = len(self.__inequality_constraints)
        iteration = 1
        while True:
            print('[*] Iteration : ', iteration)
            print('\t[*] Computing the dual coefficient matrix')
            coefficient = self.make_dual_matrix()
            print('\t[*] Computing the dual constant vector')
            constant = self.make_dual_constant_vector()

            print('\t[*] Updating the primal and dual points...')
            dual_norm = np.linalg.norm(self.__current_r_dual, 2)
            etta = self.__surrogate_duality_gaps[-1]
            self.__plots.append(etta)
            self.__dual_norms.append(dual_norm)
            print('\t[*] Dual Norm = ', dual_norm, ' Surrogate Duality Gap = ', etta)
            if dual_norm <= self.__feasibility_tolerance and etta <= self.__eta_tolerance:
                break

            delta_y = np.linalg.solve(coefficient, constant)

            delta_x = delta_y[0:50]
            delta_lambda = delta_y[50:50+m]

            self.__current_point += delta_x
            self.__current_lambda += delta_lambda
            print('\t[*] Primal and dual points updated')
            # print('\t[*] Primal point is feasible : ', self.check_primal_feasibility())
            # print('\t[*] Dual point is feasible : ', self.check_dual_feasibility())
            iteration += 1

        return self.__current_point, self.__current_lambda

    def draw_surrogate_duality_gap(self):
        fig, axs = plt.subplots(2, 1)
        columns = ['Iteration', 'Surrogate Duality Gap']
        n_rows = len(self.__plots)
        cell_text = []
        for i in range(n_rows):
            cell_text.append([i + 1, self.__plots[i]])
        axs[1].axis('tight')
        axs[1].axis('off')
        the_table = axs[1].table(cellText=cell_text, colLabels=columns, loc='center')
        axs[0].plot(self.__plots[2:], 'ro')

        plt.show()

    def draw_dual_residual_norm(self):
        fig2, axs2 = plt.subplots(2, 1)
        columns = ['Iteration', 'Dual Residual Norm']
        n_rows = len(self.__dual_norms)
        cell_text = []
        for i in range(n_rows):
            cell_text.append([i + 1, self.__dual_norms[i]])
        axs2[1].axis('tight')
        axs2[1].axis('off')
        the_table2 = axs2[1].table(cellText=cell_text, colLabels=columns, loc='center')
        axs2[0].plot(self.__dual_norms[2:], 'go')

        plt.show()
