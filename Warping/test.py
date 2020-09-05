import cv2
import time
import numpy as np

from itertools import combinations
from warp import warp as warp_func, connected_components
from utils import quatization


def main():
    start_time = time.time()
    image = cv2.imread('./view1.png')
    depth = cv2.imread('./disp1.png', 0)
    print("--- loading: %s seconds ---" % (time.time() - start_time))
    depth = 255 - depth
    start_time = time.time()

    src_pose = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    dst_pose = np.array([[1, 0, 0, 5000],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

    new_depth, new_image, check = warp_func(image, depth, src_pose, dst_pose, num_iter=5)
    connected_components(255 - check)

    print("--- warp_cpu: %s seconds ---" % (time.time() - start_time))

    cv2.imshow('image', image)
    cv2.imshow('depth', depth)
    cv2.imshow('new_image', new_image)
    new_depth = (new_depth / new_depth.max() * 255).astype(np.uint8)
    cv2.imshow('new_depth', new_depth)
    cv2.imshow('check', check)
    cv2.waitKey()


main()
