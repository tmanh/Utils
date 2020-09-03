import numpy as np

from triangle import inside_triangle
from matmul import fast_matmul, matmul_fff, matmul_fif
from utils import round
from numba import jit, cuda


# @jit(nopython=True)
def generate_triangles_down_right(row_triangles, col_triangles, width, height):
    triangles = np.zeros((len(row_triangles), 6), dtype=np.int32)
    k = 0
    for i in range(len(row_triangles)):
        triangles[k, 0] = row_triangles[i]
        triangles[k, 1] = col_triangles[i]
        
        if row_triangles[i] + 1 < height:
            triangles[k, 2] = row_triangles[i] + 1
        else:
            continue
        triangles[k, 3] = col_triangles[i]
        
        triangles[k, 4] = row_triangles[i]

        if col_triangles[i] + 1 < width:
            triangles[k, 5] = col_triangles[i] + 1
        else:
            continue

        k += 1
    return triangles


# @jit(nopython=True)
def generate_triangles_up_left(row_triangles, col_triangles, width, height):
    triangles = np.zeros((len(row_triangles), 6), dtype=np.int32)
    k = 0
    for i in range(len(row_triangles)):
        triangles[k, 0] = row_triangles[i]
        triangles[k, 1] = col_triangles[i]
        
        if row_triangles[i] - 1 >= 0:
            triangles[k, 2] = row_triangles[i] - 1
        else:
            continue
        triangles[k, 3] = col_triangles[i]
        
        triangles[k, 4] = row_triangles[i]

        if col_triangles[i] - 1 >= 0:
            triangles[k, 5] = col_triangles[i] - 1
        else:
            continue

        k += 1
    return triangles


def generate_triangles(quantizied):
    triangles = np.zeros((0, 6), dtype=np.int32)
    # x - o
    # o
    flag = quantizied == 0
    flag_border = np.ones_like(flag)
    flag_border[:, 0] = False
    flag_border[:, flag_border.shape[1] - 1] = False
    flag_border[0, :] = False
    flag_border[flag_border.shape[0] - 1, :] = False

    down_flag = np.roll(flag, -1, axis=0)
    right_flag = np.roll(flag, -1, axis=1)
    flag_1 = ((flag & down_flag) & right_flag) & flag_border
    rows, cols = np.where(flag_1 == True)

    down_right = generate_triangles_down_right(rows, cols, quantizied.shape[1], quantizied.shape[0])
    triangles = np.concatenate([triangles, down_right], axis=0)

    # . - o
    # o - x
    up_flag = np.roll(flag, 1, axis=0)
    left_flag = np.roll(flag, 1, axis=1)
    flag_2 = ((flag & up_flag) & left_flag) & flag_border
    rows, cols = np.where(flag_2 == True)

    up_left = generate_triangles_up_left(rows, cols, quantizied.shape[1], quantizied.shape[0])
    triangles = np.concatenate([triangles, up_left], axis=0)

    return triangles


@jit('void(float32,float32,float32,float32,float32,float32,float32,float32,float32,float32[:,:])')
def create_loc_matrix(t0, t1, t2, t3, t4, t5, d1, d2, d3, locs):
    locs[0, 0] = t1
    locs[0, 1] = t3
    locs[0, 2] = t5
    locs[1, 0] = t0
    locs[1, 1] = t2
    locs[1, 2] = t4
    locs[2, 0] = d1
    locs[2, 1] = d2
    locs[2, 2] = d3


@jit('void(int32[:,:],float32[:,:],float32[:,:],float32[:,:],uint8[:,:],int64,int64,int64)')
def warp_cpu(triangles, depth, inv_src_pose, dst_pose, new_depth, iter_num, height, width):
    # import cv2
    auto = False
    for i in range(iter_num):
        locs = np.ones((4, 3), dtype=np.float32)
        create_loc_matrix(triangles[i, 0], triangles[i, 1], triangles[i, 2],
                          triangles[i, 3], triangles[i, 4], triangles[i, 5],
                          depth[triangles[i, 0], triangles[i, 1]],
                          depth[triangles[i, 2], triangles[i, 3]],
                          depth[triangles[i, 4], triangles[i, 5]], locs)
        warped_locs = np.zeros((4, 3), dtype=np.float32)
        matmul_fff(inv_src_pose, locs, warped_locs)
        matmul_fff(dst_pose, warped_locs, warped_locs)

        for j in range(3):
            x = int(warped_locs[0, j] / warped_locs[3, j])
            y = int(warped_locs[1, j] / warped_locs[3, j])
            if 0 <= x and x < width and 0 <= y and y < height:
                if new_depth[y, x] > round(warped_locs[2, j] / warped_locs[3, j]) or new_depth[y, x] == 0:
                    new_depth[y, x] = round(warped_locs[2, j] / warped_locs[3, j])

        X_out, Y_out = inside_triangle(
            warped_locs[0, 0] / warped_locs[3, 0],
            warped_locs[0, 1] / warped_locs[3, 1],
            warped_locs[0, 2] / warped_locs[3, 2],
            warped_locs[1, 0] / warped_locs[3, 0],
            warped_locs[1, 1] / warped_locs[3, 1],
            warped_locs[1, 2] / warped_locs[3, 2])

        for j in range(len(X_out)):
            x = X_out[j]
            y = Y_out[j]
            dis1 = abs(x - warped_locs[0, 0] / warped_locs[3, 0]) + abs(y - warped_locs[1, 0] / warped_locs[3, 0])
            dis2 = abs(x - warped_locs[0, 1] / warped_locs[3, 1]) + abs(y - warped_locs[1, 1] / warped_locs[3, 1])
            dis3 = abs(x - warped_locs[0, 2] / warped_locs[3, 2]) + abs(y - warped_locs[1, 2] / warped_locs[3, 2])
            d1 = warped_locs[2, 0] / warped_locs[3, 0]
            d2 = warped_locs[2, 1] / warped_locs[3, 1]
            d3 = warped_locs[2, 2] / warped_locs[3, 2]
            dis_t = dis1 + dis2 + dis3
            d = d1 * dis1 / dis_t + d2 * dis2 / dis_t + d3 * dis3 / dis_t
            if 0 <= x and x < width and 0 <= y and y < height:
                if new_depth[y, x] > d or new_depth[y, x] == 0:
                    new_depth[y, x] = round(d)


@cuda.jit
def warp_gpu(triangles, depth, src_pose, dst_pose):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        # note that calling a numba.jit function from CUDA automatically
        # compiles an equivalent CUDA device function!
        bin_number = compute_bin(x[i], nbins, xmin, xmax)

        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)
