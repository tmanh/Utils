import cv2
import time
import numpy as np

from itertools import combinations
from warp import generate_triangles, warp_cpu
from bilateral import sparse_bilateral_filtering


def quatization(depth):
    div = 32
    quantized = depth // div * div + div // 2
    return quantized, [i * div + div // 2 for i in range(255 // div)]


def main():
    image = cv2.imread('./view1.png')
    depth = cv2.imread('./disp1.png', 0)
    depth = 255 - depth

    vis_photos, bilateral = sparse_bilateral_filtering(depth, image, num_iter=5)
    bilateral = bilateral[-1]
    xxx = bilateral - bilateral.min()
    xxx = xxx / xxx.max()
    
    horizon_check = (abs(xxx - np.roll(xxx, 1, 0)) > 0.04).astype(np.uint8) * 255
    vertical_check = (abs(xxx - np.roll(xxx, 1, 1)) > 0.04).astype(np.uint8) * 255
    check = np.maximum(horizon_check, vertical_check)

    src_pose = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    dst_pose = np.array([[1, 0.3, 1, 5],
                         [0, 1, 0.5, 4],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)  
    
    start_time = time.time()
    triangles = generate_triangles(check)
    print("--- %s seconds ---" % (time.time() - start_time))
    new_depth = np.zeros(depth.shape, dtype=np.uint8)
    start_time = time.time()
    warp_cpu(triangles, depth.astype(np.float32), np.linalg.inv(src_pose),
             dst_pose, new_depth, triangles.shape[0], depth.shape[0], depth.shape[1])
    print("--- %s seconds ---" % (time.time() - start_time))
    # cv2.imshow('image', image)
    cv2.imshow('depth', depth)
    cv2.imshow('new_depth', new_depth)
    cv2.imshow('check', check)
    cv2.waitKey()


main()
