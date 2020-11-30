from line_search.wolfe_conditions import Wolfe


class Backtracking:
    """
    This class implements backtracking algorithm to find a proper alpha
    """

    __func = None
    __p = None
    __function = None
    __descent_direction = None
    __current_point = None
    __constant1 = None
    __constant2 = None

    def __init__(self, func, descent_direction, current_point, c1, c2, p=0.9):
        self.__func = func.get_function()
        self.__function = func
        self.__descent_direction = descent_direction
        self.__current_point = current_point
        self.__p = p
        self.__constant1 = c1
        self.__constant2 = c2

    def get_alpha(self):
        wolfe = Wolfe(self.__function, self.__current_point, self.__descent_direction,
                      self.__constant1, self.__constant2)
        step_lengths = wolfe.get_acceptable_alpha()

        if len(step_lengths) == 0:
            return None
        alpha0 = max(step_lengths)

        condition = wolfe.check_first_condition(self.__current_point, alpha0)

        while condition is False:
            alpha0 = self.__p * alpha0
            condition = wolfe.check_first_condition(self.__current_point, alpha0)

        return alpha0
