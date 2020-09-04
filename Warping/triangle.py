import numpy as np
import numba as nb

from utils import round
from numba import jit


@jit('void(float32,float32,float32,float32,float32,float32,float32[:],float32[:])')
def create_list_coordinates(x1, x2, x3, y1, y2, y3, xs, ys):
    xs[0] = x1
    xs[1] = x2
    xs[2] = x3
    ys[0] = y1
    ys[1] = y2
    ys[2] = y3


@jit('void(int32,int32,int32[:])')
def arange(min_value, max_value, output):
    for i in range(min_value, max_value + 1):
        output[i - min_value] = i


@jit('void(int32[:],int32[:],int32[:,:],int32[:,:])')
def meshgrid(x_range, y_range, X, Y):
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            X[j, i] = x_range[i]
            Y[j, i] = y_range[j]


@jit('float32(float32,float32,int32)')
def fit_line(a, b, x):
    return a * x + b
    

@jit('float32[:,:](float32,float32,int32[:,:])')
def fit_lines(a, b, x):
    return a * x + b


@jit('int32(boolean[:,:],int32,int32)')
def count_true(triangle, len_i, len_j):
    count = int(0)
    for i in range(len_i):
        for j in range(len_j):
            if triangle[i, j]:
                count += 1
    return count


@jit('void(boolean[:,:],int32[:,:],int32[:,:],int32[:],int32[:],int32,int32)')
def get_coordinates(triangle, X, Y, X_out, Y_out, len_i, len_j):
    k = int(0)
    for i in range(len_i):
        for j in range(len_j):
            if triangle[i, j]:
                X_out[k] = X[i, j]
                Y_out[k] = Y[i, j]
                k += 1


@jit('Tuple((int32[:],int32[:]))(float32,float32,float32,float32,float32,float32)')
def inside_triangle(x1, x2, x3, y1, y2, y3):
    xs = np.ones(3, dtype=np.float32)
    ys = np.ones(3, dtype=np.float32)
    create_list_coordinates(x1, x2, x3, y1, y2, y3, xs, ys)

    xc = np.mean(xs)
    yc = np.mean(ys)

    for i in range(3):
        if int(xc) == int(xs[i]) and int(yc) == int(ys[i]):
            return xs.astype(np.int32), ys.astype(np.int32)

    # The possible range of coordinates that can be returned
    len_x = int(int(np.max(xs) + 1) - int(np.min(xs)))
    x_range = np.zeros(len_x, dtype=np.int32)
    arange(int(np.min(xs)), int(np.max(xs) + 1), x_range)
    len_y = int(int(np.max(ys) + 1) - int(np.min(ys)))
    y_range = np.zeros(len_y, dtype=np.int32)
    arange(int(np.min(ys)), int(np.max(ys) + 1), y_range)

    X = np.zeros((len_y, len_x), dtype=np.int32)
    Y = np.zeros((len_y, len_x), dtype=np.int32)

    # Set the grid of coordinates on which the triangle lies. The centre of the
    # triangle serves as a criterion for what is inside or outside the triangle.
    meshgrid(x_range, y_range, X, Y)

    # From the array 'triangle', points that lie outside the triangle will be
    # set to 'False'.
    triangle = np.ones((len_y, len_x), dtype=nb.boolean)  # nb.boolean
    on_edges = np.ones((len_y, len_x), dtype=nb.boolean)

    # if 3 points form a triangle
    for i in range(3):
        ii = (i+1) % 3
        if xs[i]==xs[ii]:
            if xc > xs[i]:
                include = X > xs[i] * 0
            else:
                include = X < xs[i] * 0
            check_on_edges = np.abs(X - xs[i]) <= 1
        else:
            a = (ys[ii] - ys[i]) / (xs[ii] - xs[i])
            b = ys[i] - xs[i] * (ys[ii] - ys[i]) / (xs[ii] - xs[i])
            points_on_line = fit_lines(a, b, X)

            if yc - fit_line(a, b, xc) > 0:
                include = Y > points_on_line
            else:
                include = Y < points_on_line

            check_on_edges = np.abs(Y - points_on_line) <= 1

        on_edges |= check_on_edges
        triangle *= include
    
    triangle = triangle | on_edges

    # Output: 2 arrays with the x- and y- coordinates of the points inside the
    # triangle.
    count = count_true(triangle, len_y, len_x)
    X_tmp = np.zeros(count, dtype=np.int32)
    Y_tmp = np.zeros(count, dtype=np.int32)
    get_coordinates(triangle, X, Y, X_tmp, Y_tmp, len_y, len_x)
    
    return X_tmp, Y_tmp
