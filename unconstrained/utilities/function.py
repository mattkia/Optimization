import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Function:
    """
    This class implements the definition of a function, which can be either 1 dimensional or 2 dimensional depending on
    the definition of the lambda expression of the function
    @:param __func : must be a lambda expression defining the function
    @:param __domain1 : must be an array representing the function domain for the first variable
    @:param __domain2 : must be an array representing the function domain for the second variable (if the function is
    single variable this variable must be left unfilled)
    @:param __func_values : stores the values of the function at each point in the domain
    @:param __dimension : stores the dimension of the function
    Constructor of the class must be filled with the function description and at least domain1
    """
    __func = None
    __func_values = list()
    __domain1 = list()
    __domain2 = list()
    __dimension = None

    def __init__(self, func=None, domain1=None, domain2=None):
        self.__domain1 = domain1
        self.__domain2 = domain2
        self.__func = func
        self.__dimension = 1 if domain2 is None else 2
        if self.__dimension == 1:
            self.calculate_function()
        else:
            self.calculate_2d_function()

    def get_function(self):
        return self.__func

    def get_domain1(self):
        return self.__domain1

    def get_domain2(self):
        return self.__domain2

    def get_dimension(self):
        return self.__dimension

    def draw_function(self):
        """
        This method can only draw a 1D and 2D functions
        :return: None, shows the graph of the function
        """
        if self.__dimension == 1:
            plt.plot(self.__domain1, self.__func_values)
            plt.title('Graph of the function')
            plt.xlabel('X')
            plt.ylabel('f(X)')
            plt.show()
        elif self.__dimension == 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('f(X1,X2)')
            x, y = np.meshgrid(self.__domain1, self.__domain2)
            z = self.__func(x, y)
            surf = ax.plot_surface(x, y, z, cmap=cm.Spectral)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()
        else:
            raise Exception('Wrong input format')

    def calculate_function(self):
        """
        This method calculates the values of the function corresponding to each point in the specified domain of
        the function.
        :return: None, fills the func_values property with the values of the function at each domain point
        """
        if self.__dimension == 1:
            points_numbers = len(self.__domain1)
            for i in range(points_numbers):
                val = self.__func(self.__domain1[i])
                self.__func_values.append(val)

    def calculate_2d_function(self):
        """
        This method calculates the values of the function corresponding to each point in the 2D grid of domain
        :return: None, fills the func_values property with the values of the function at each grid point
        """
        x, y = np.meshgrid(self.__domain1, self.__domain2)
        self.__func_values = self.__func(x, y)

