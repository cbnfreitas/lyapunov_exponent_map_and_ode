import unittest

from numpy import *

from henon_map import HenonMap
from lorenz_map import LorenzMap
from lyapunov_exponents import lyapunov_exponent
from numpy.linalg import *

# Ref: Alligood, Kathleen T., Tim D. Sauer, and James A. Yorke. Chaos. Springer New York, 1996.
HENON_MAP = HenonMap()
HENON_MAP_REFERENCE_SOLUTION = array([-1.61, 0.41])
TEST_TOLERANCE = 0.01
x_points, y_points = mgrid[-1:1:0.25, -1:1:0.25]
HENON_MAP_INITIAL_CONDITIONS = column_stack((x_points.ravel(), y_points.ravel()))

LORENZ_MAP = LorenzMap()
LORENZ_MAP_REFERENCE_SOLUTION = array([-14.57, 0, 0.90])
LORENZ_MAP_INITIAL_CONDITION = array([-5.76,  2.27,  32.82])


class LyapunovExponentsTestCase(unittest.TestCase):
    def henon_map_test(_):
        l = lyapunov_exponent(HENON_MAP, multiple_initial_conditions=HENON_MAP_INITIAL_CONDITIONS)
        _.assertEqual(norm(HENON_MAP_REFERENCE_SOLUTION - l, inf) < TEST_TOLERANCE, True)

    def lorenz_map_test(_):
        l = lyapunov_exponent(LORENZ_MAP, single_initial_condition=LORENZ_MAP_INITIAL_CONDITION)
        _.assertEqual(norm(LORENZ_MAP_REFERENCE_SOLUTION - l, inf) < TEST_TOLERANCE, True)


if __name__ == '__main__':
    unittest.main()
