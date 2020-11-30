from constrained.constrained_newton import ConstrainedNewton
from constrained.problem import Problem
from constrained.barrier import Barrier
from constrained.primal_dual import PrimalDualInteriorPoint
from scipy import random
import numpy as np


quadratic_matrices = []
q_vectors = []
inequality_coefficient_matrices = []
inequality_constant_vectors = []
initial_points = []
initial_lambdas = []
initial_nus = []
functions = []
problems = []
barriers = []
primal_duals = []
mus = []

# Creating the positive definite matrices
for i in range(5):
    a = random.rand(50, 50)
    b = np.dot(a, a.transpose())
    quadratic_matrices.append(b)

# Creating the inequality coefficient matrix , constant vector , and initial strictly feasible points
for i in range(5):
    a = random.rand(100, 50)
    point = random.rand(50, 1)
    initial_points.append(point)
    delta = np.ones((100, 1))
    b = np.matmul(a, point) + delta
    inequality_coefficient_matrices.append(a)
    inequality_constant_vectors.append(b)

# Creating the q vector of the quadratic function
for i in range(5):
    vector = random.rand(1, 50)
    q_vectors.append(vector)

# Constructing the functions
for i in range(5):
    func = lambda x: (1 / 2) * np.matmul(np.matmul(x.reshape(1, 50), quadratic_matrices[i]), x) + \
                     np.matmul(q_vectors[i], x.reshape(50, 1))
    functions.append(func)

# Constructing the problems
for i in range(5):
    inequalities = []
    initial_lambda = np.zeros((100, 1))
    for j in range(100):
        inequalities.append(lambda x: np.matmul(inequality_coefficient_matrices[i][j], x.reshape(50, 1)) -
                                      inequality_constant_vectors[i][j])
    for j in range(100):
        initial_lambda[j] = - (1 / inequalities[j](initial_points[i]))
    initial_lambdas.append(initial_lambda)
    problem = Problem(functions[i], inequalities, None, None)
    problems.append(problem)

for i in range(5):
    mus.append(np.random.uniform(1, 3))

# Constructing the barrier problems
for i in range(5):
    barrier = Barrier(problems[i], initial_points[i], 10e4, mus[i], 10e-5)
    barriers.append(barrier)

# Constructing the primal-dual problems
for i in range(5):
    primal_dual = PrimalDualInteriorPoint(problems[i], initial_points[i], initial_lambdas[i], 10e-3, 10e-3, mus[i])
    primal_duals.append(primal_dual)

# Running the Problems
# for i in range(5):
#     print(barriers[i].run())
#     barriers[i].draw_duality_gaps()

# Running the problems
# for i in range(5):
#     print(primal_duals[i].run())
#     primal_duals[i].draw_surrogate_duality_gap()
#     primal_duals[i].draw_dual_residual_norm()

print(primal_duals[4].run())
primal_duals[4].draw_surrogate_duality_gap()
primal_duals[4].draw_dual_residual_norm()
