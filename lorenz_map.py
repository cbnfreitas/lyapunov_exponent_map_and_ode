from scipy import integrate
from numpy import *


class LorenzMap:
    """ The Lorenz map corresponds to advancing one unit of time over the integral curves of the Lorenz System.
        This class holds the parameters of the Lorenz Map and evaluate it along its directional derivative,
        which is computed via variational equation.
        The default parameters are chosen as the canonical ones in the initialization.

        It instantiates a callable object f_df, in such a way that f_df(xyz, w)
        returns two values f(xyz) and df(xyz, w), where
        f(xyz) is the solution of the Lorenz system starting at xyz after one unit of time and
        df(xyz, w) is the solution of the Lorenz variational equations starting at (xyz, w).
    """

    def __init__(_, sigma=10, rho=28, beta=8 / 3, h0=0.01):
        _.sigma, _.rho, _.beta = sigma, rho, beta
        _.h0 = h0

    @staticmethod
    def pack_variables(xyz, w):
        return concatenate((xyz, reshape(w, 9)), axis=0)

    @staticmethod
    def unpack_variables(xyzw):
        return xyzw[0:3], reshape(xyzw[3::], (3, 3))

    def variational_equation(_, xyzw, t=None):
        xyz, w = _.unpack_variables(xyzw)
        x, y, z = xyz

        dot_xyz = array([_.sigma * (-x + y),
                         x * (_.rho - z) - y,
                         x * y - _.beta * z])

        dot_w = array([[ -_.sigma, _.sigma,       0],
                       [_.rho - z,      -1,      -x],
                       [        y,       x, -_.beta]]) @ w

        return _.pack_variables(dot_xyz, dot_w)

    def __call__(_, xyz, w):
        xyzw = _.pack_variables(xyz, w)
        next_xyzw = integrate.odeint(_.variational_equation, xyzw, array([0, 1]), h0=_.h0)
        return _.unpack_variables(next_xyzw[1])
