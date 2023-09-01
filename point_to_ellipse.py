
# @sharnett
# point_to_ellipse
# https://github.com/sharnett/point_to_ellipse


import math
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.4);

import itertools
from numpy import sin, cos, pi

def step(t, a, b, px, py):
    x, y = a*cos(t), b*sin(t)

    ex = (a*a - b*b) * cos(t)**3 / a
    ey = (b*b - a*a) * sin(t)**3 / b

    rx, ry = x - ex, y - ey
    qx, qy = px - ex, py - ey
    r, q = math.hypot(ry, rx), math.hypot(qy, qx)

    delta_c = r * math.asin((rx*qy - ry*qx)/(r*q))
    delta_t = delta_c / math.sqrt(a*a + b*b - x*x - y*y)
    return delta_t

TOL = 1e-3
T_MIN, T_START, T_MAX = 0, pi/4, pi/2
MAX_IT = 10

def f(t, a, b, x, y):
    return (x - a*cos(t))**2 + (y - b*sin(t))**2


def df(t, a, b, x, y):
    c2 = a**2 - b**2
    return 2*(a*x*sin(t) - b*y*cos(t) - c2*sin(t)*cos(t))


def d2f(t, a, b, x, y):
    c2 = a**2 - b**2
    return 2*(a*x*cos(t) + b*y*sin(t) - c2*cos(2*t))


def newton(a, b, x, y):
    if (x, y) == (0, 0): return min(a, b)
    x, y = abs(x), abs(y)
    t = T_START
    for i in range(MAX_IT):
        t_step = -df(t, a, b, x, y)/d2f(t, a, b, x, y)
        t = np.clip(t + t_step, T_MIN, T_MAX)
        if abs(t_step) < TOL: break
    else:
        pass
        # print('Distance failed to converge in %d iterations' % MAX_IT)
        # raise RuntimeError('Newton failed to converge in %d iterations' % MAX_IT)
        # raise RuntimeWarning('Newton failed to converge in %d iterations' % MAX_IT)
    return f(t, a, b, x, y)**.5, i

def new_method(a, b, px, py):
    if (px, py) == (0, 0): return min(a, b)
    px, py = abs(px), abs(py)
    t = T_START
    for i in range(MAX_IT):
        t_step = step(t, a, b, px, py)
        t = np.clip(t + t_step, T_MIN, T_MAX)
        if abs(t_step) < TOL: break
    else:
        pass
        # print('Distance failed to converge in %d iterations' % MAX_IT)
        # raise RuntimeWarning('New method failed to converge in %d iterations' % MAX_IT)
    return f(t, a, b, px, py)**.5, i


SPECIAL_ITS = 3
def hybrid(a, b, x, y):
    if (x, y) == (0, 0): return min(a, b)
    x, y = abs(x), abs(y)
    t = T_START
    for i in range(MAX_IT):
        t_step = (step(t, a, b, x, y) if i < SPECIAL_ITS
                  else -df(t, a, b, x, y)/d2f(t, a, b, x, y))
        t = np.clip(t + t_step, T_MIN, T_MAX)
        if abs(t_step) < TOL:
            return f(t, a, b, x, y)**.5, i
    else:
        pass
        # print('Distance failed to converge in %d iterations' % MAX_IT)
        # raise RuntimeWarning('Failed to converge in %d iterations' % MAX_IT)
    return f(t, a, b, x, y)**.5, i

