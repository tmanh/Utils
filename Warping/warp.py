import numpy as np
from numba import jit, cuda

from bilateral import sparse_bilateral_filtering
from triangle import inside_triangle
from matmul import fast_matmul, matmul_fff, matmul_fif
from utils import round


@jit('int32[:,:](int64[:],int64[:],int64,int64)')
def generate_triangles_down_right(row_triangles, col_triangles, width, height):
    triangles = np.zeros((len(row_triangles), 6), dtype=np.int32)
    for i in range(len(row_triangles)):
        triangles[i, 0] = row_triangles[i]
        triangles[i, 1] = col_triangles[i]
        
        triangles[i, 2] = row_triangles[i] + 1
        triangles[i, 3] = col_triangles[i]
        
        triangles[i, 4] = row_triangles[i]
        triangles[i, 5] = col_triangles[i] + 1

        i += 1
    return triangles


@jit('int32[:,:](int64[:],int64[:],int64,int64)')
def generate_triangles_up_left(row_triangles, col_triangles, width, height):
    triangles = np.zeros((len(row_triangles), 6), dtype=np.int32)
    
    for i in range(len(row_triangles)):
        triangles[i, 0] = row_triangles[i]
        triangles[i, 1] = col_triangles[i]
        
        triangles[i, 2] = row_triangles[i] - 1
        triangles[i, 3] = col_triangles[i]
        
        triangles[i, 4] = row_triangles[i]
        triangles[i, 5] = col_triangles[i] - 1
    return triangles


def generate_triangles(quantizied):
    triangles = np.zeros((0, 6), dtype=np.int32)

    flag = quantizied == 0

    flag_border = np.ones_like(flag)
    flag_border[:, 0] = False
    flag_border[:, flag_border.shape[1] - 1] = False
    flag_border[0, :] = False
    flag_border[flag_border.shape[0] - 1, :] = False

    # x - o
    # o
    down_flag = np.roll(flag, -1, axis=0)
    right_flag = np.roll(flag, -1, axis=1)
    flag_1 = ((flag & down_flag) & right_flag) & flag_border
    rows, cols = np.where(flag_1 == True)

    # generate triangles
    down_right = generate_triangles_down_right(rows, cols, quantizied.shape[1], quantizied.shape[0])
    triangles = np.concatenate([triangles, down_right], axis=0)

    # . - o
    # o - x
    up_flag = np.roll(flag, 1, axis=0)
    left_flag = np.roll(flag, 1, axis=1)
    flag_2 = ((flag & up_flag) & left_flag) & flag_border
    rows, cols = np.where(flag_2 == True)

    # generate triangles
    up_left = generate_triangles_up_left(rows, cols, quantizied.shape[1], quantizied.shape[0])
    triangles = np.concatenate([triangles, up_left], axis=0)

    return triangles


@jit('void(float32,float32,float32,float32,float32,float32,float32,float32,float32,float32[:,:])')
def create_loc_matrix(t0, t1, t2, t3, t4, t5, d1, d2, d3, locs):
    locs[0, 0] = t1 * d1
    locs[0, 1] = t3 * d2
    locs[0, 2] = t5 * d3
    locs[1, 0] = t0 * d1
    locs[1, 1] = t2 * d2
    locs[1, 2] = t4 * d3
    locs[2, 0] = d1
    locs[2, 1] = d2
    locs[2, 2] = d3


def create_loc_matrix_from_depth(depth, height, width):
    y = np.arange(0, height).repeat(width, 0).reshape((1, height, width))                    # columns
    x = np.arange(0, width).reshape(1, width).repeat(height, 0).reshape((1, height, width))  # rows
    z = depth.reshape((1, height, width))
    ones = np.ones((1, height, width))

    pos_matrix = np.concatenate([x * z, y * z, z, ones], axis=0)

    return pos_matrix


def pre_warp(pos_matrix, inv_src_pose, dst_pose, height, width):
    world_p = inv_src_pose @ pos_matrix.reshape((4, -1))
    warped_locs = dst_pose @ world_p
    return (warped_locs.reshape((4 , height, width)) + 1e-7).astype(np.float32)


@jit('void(int32[:,:],float32[:,:,:],float32[:,:],float32[:,:],int64,int64,int64)')
def warp_cpu(triangles, warped_locs, depth, new_depth, iter_num, height, width):
    warped_locs[0, :] = warped_locs[0, :] / warped_locs[2, :]
    warped_locs[1, :] = warped_locs[1, :] / warped_locs[2, :]
    for i in range(iter_num):
        warped_loc = np.zeros((4, 3), dtype=np.float32)
        for j in range(3):
            warped_loc[:, j] = warped_locs[:, triangles[i, j * 2], triangles[i, j * 2 + 1]]

        for j in range(3):
            x = int(warped_loc[0, j])
            y = int(warped_loc[1, j])
            if 0 <= x and x < width and 0 <= y and y < height:
                if new_depth[y, x] > warped_loc[2, j] or new_depth[y, x] == 0:
                    new_depth[y, x] = warped_loc[2, j]

        X_out, Y_out = inside_triangle(warped_loc[0, 0], warped_loc[0, 1], warped_loc[0, 2],
            warped_loc[1, 0], warped_loc[1, 1], warped_loc[1, 2])

        for j in range(len(X_out)):
            x = X_out[j]
            y = Y_out[j]
            
            dis1 = abs(x - warped_loc[0, 0]) + abs(y - warped_loc[1, 0])
            dis2 = abs(x - warped_loc[0, 1]) + abs(y - warped_loc[1, 1])
            dis3 = abs(x - warped_loc[0, 2]) + abs(y - warped_loc[1, 2])
            
            d1 = warped_loc[2, 0]
            d2 = warped_loc[2, 1]
            d3 = warped_loc[2, 2]
            
            dis_t = dis1 + dis2 + dis3
            
            d = d1 * dis1 / dis_t + d2 * dis2 / dis_t + d3 * dis3 / dis_t
            
            if 0 <= x and x < width and 0 <= y and y < height:
                if new_depth[y, x] > d or new_depth[y, x] == 0:
                    new_depth[y, x] = round(d)


def warp(image, depth, src_pose, dst_pose, num_iter=5):
    vis_photos, bilateral = sparse_bilateral_filtering(depth, image, num_iter=num_iter)
    bilateral = bilateral[-1]

    norm_bilateral = (bilateral - bilateral.min()) / (bilateral.max() - bilateral.min())
    xxx = bilateral / bilateral.max()

    horizon_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 0)) > 0.03).astype(np.uint8) * 255
    vertical_check = (abs(norm_bilateral - np.roll(norm_bilateral, 1, 1)) > 0.03).astype(np.uint8) * 255
    check = np.maximum(horizon_check, vertical_check)

    triangles = generate_triangles(check)
    
    new_depth = np.zeros(depth.shape, dtype=np.float32)
    pos_matrix = create_loc_matrix_from_depth(depth, depth.shape[0], depth.shape[1])
    warped_locs = pre_warp(pos_matrix, np.linalg.inv(src_pose), dst_pose, depth.shape[0], depth.shape[1])
    warp_cpu(triangles, warped_locs, depth.astype(np.float32), new_depth, triangles.shape[0], depth.shape[0], depth.shape[1])

    return new_depth, check
