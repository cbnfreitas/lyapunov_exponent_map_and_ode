from numpy import *


class HenonMap:
    """ Class to hold the parameters of the Henon Map and evaluate it along its directional derivative.
        The default parameters are chosen as the canonical ones in the initialization.

        It instantiates a callable object f_df, in such a way that f_df(xy, w)
        returns two values f(xy) and df(xy, w), where
        f(xy) is the Henon map evaluated at the point xy and
        df(xy, w) is the differential of the Henon evaluated at xy in the direction of w.
    """
    def __init__(_, a=1.4, b=0.3):
        _.a, _.b = a, b

    def f(_, xy):
        x, y = xy
        return array([_.a - x ** 2 + _.b * y, x])

    def df(_, xy, w):
        x, y = xy
        j = array([[-2 * x, _.b],
                   [1, 0]])
        return j @ w

    def __call__(_, xy, w):
        return _.f(xy), _.df(xy, w)
