import cv2
import time
import numpy as np

from itertools import combinations
from warp import generate_triangles, warp_cpu, create_loc_matrix_from_depth, pre_warp
from bilateral import sparse_bilateral_filtering


def quatization(depth):
    div = 32
    quantized = depth // div * div + div // 2
    return quantized, [i * div + div // 2 for i in range(255 // div)]


def main():
    start_time = time.time()
    image = cv2.imread('./view1.png')
    depth = cv2.imread('./disp1.png', 0)
    print("--- loading: %s seconds ---" % (time.time() - start_time))
    depth = 255 - depth
    start_time = time.time()
    vis_photos, bilateral = sparse_bilateral_filtering(depth, image, num_iter=5)
    bilateral = bilateral[-1]
    xxxx = (bilateral - bilateral.min()) / (bilateral.max() - bilateral.min())
    xxx = bilateral / bilateral.max()
    
    horizon_check = (abs(xxxx - np.roll(xxxx, 1, 0)) > 0.03).astype(np.uint8) * 255
    vertical_check = (abs(xxxx - np.roll(xxxx, 1, 1)) > 0.03).astype(np.uint8) * 255
    check = np.maximum(horizon_check, vertical_check)

    src_pose = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    dst_pose = np.array([[1, 0, 0, 5000],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)  
    print("--- preprocess: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    triangles = generate_triangles(check)
    print("--- generate_triangles: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    new_depth = np.zeros(depth.shape, dtype=np.float32)

    # """
    pos_matrix = create_loc_matrix_from_depth(depth, depth.shape[0], depth.shape[1])
    warped_locs = pre_warp(pos_matrix, np.linalg.inv(src_pose), dst_pose, depth.shape[0], depth.shape[1])

    warp_cpu(triangles, warped_locs, depth.astype(np.float32), new_depth, triangles.shape[0], depth.shape[0], depth.shape[1])
    print("--- warp_cpu: %s seconds ---" % (time.time() - start_time))
    # """

    cv2.imshow('depth', depth)
    new_depth = (new_depth / new_depth.max() * 255).astype(np.uint8)
    cv2.imshow('new_depth', new_depth)
    cv2.imshow('check', check)
    cv2.waitKey()


main()
