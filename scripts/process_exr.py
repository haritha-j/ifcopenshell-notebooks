'''
Convert EXR images into point clouds


MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
from open3d import *


def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_list')
    parser.add_argument('intrinsics_file')
    parser.add_argument('output_dir')

    args = parser.parse_args()


    # with open(args.list_file) as file:
    #     model_list = file.read().splitlines()
    num_scans = 16
    model_list = os.listdir(args.model_list)

    intrinsics = np.loadtxt(args.intrinsics_file)
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    depth_dir = os.path.join(args.output_dir, 'depth')
    pcd_dir = os.path.join(args.output_dir, 'pcd')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    for model_id in model_list:
        element_name = model_id.split(".")[0]
        exr_path = os.path.join(args.output_dir, 'exr', model_id)
        pose_path = os.path.join(args.output_dir, 'pose', element_name+'.txt')

        depth = read_exr(exr_path, height, width)
        depth_img = geometry.Image(np.uint16(depth * 1000))
        io.write_image(os.path.join(depth_dir, element_name+'.png'), depth_img)

        pose = np.loadtxt(pose_path)
        points = depth2pcd(depth, intrinsics, pose)
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(points)
        print(element_name, pcd.points)
        io.write_point_cloud(os.path.join(pcd_dir, element_name+'.pcd'), pcd)
