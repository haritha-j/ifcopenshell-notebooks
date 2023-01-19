import math
import random
import time
import torch
import numpy as np
from chamferdist import ChamferDistance
import torch.nn.functional as F

from src.visualisation import get_direction_from_trig
from src.geometry import vector_norm


def get_chamfer_dist_single(src, tgt):
    src, tgt = torch.tensor([src]).cuda().float(), torch.tensor([tgt]).cuda().float()
    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(src, tgt, bidirectional=True)
    return (bidirectional_dist.detach().cpu().item())


def get_chamfer_loss(preds_tensor, src_pcd_tensor):
    target_pcd_list = []
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    preds_list = preds_tensor.cpu().detach().numpy()
    #t1 = time.perf_counter()
    for preds in preds_list:
        target_pcd_list.append(generate_elbow_cloud(preds))
    target_pcd_tensor = torch.tensor(target_pcd_list).float().cuda()
    #t2 = time.perf_counter()

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(target_pcd_tensor, src_pcd_tensor, bidirectional=True)
    #t3 = time.perf_counter()
    #print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


def get_chamfer_loss_tensor(preds_tensor, src_pcd_tensor):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
    #t2 = time.perf_counter()
    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(target_pcd_tensor, src_pcd_tensor, bidirectional=True)
    #t3 = time.perf_counter()
    #print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


# generate points on surface of elbow
def generate_elbow_cloud(preds):
    # read params
    r, x, y = preds[0], preds[1], preds[2], 
    d = get_direction_from_trig(preds, 8)
    a = math.atan2(preds[3], preds[4])
    p = [preds[5], preds[6], preds[7]]
    
    # get new coordinate frame of elbow
    old_z = (0., 0., 1.)
    x_axis = vector_norm(np.cross(d, old_z))
    y_axis = vector_norm(np.cross(d, x_axis))
    
    # compute transformation
    rot_mat = np.transpose(np.array([x_axis, y_axis, d]))
    t = np.array([[p[0]], [p[1]], [p[2]]])
    
    trans_mat = np.vstack((np.hstack((rot_mat, t)), 
                          [0., 0., 0., 1.]))    
    
    # compute axis
    theta = math.atan2(x, y)
    original_axis_dir = [math.cos(theta), -1*math.sin(theta), 0., 0.]
    transformed_axis_dir = np.matmul(trans_mat, np.array(original_axis_dir))[:-1]
    #print(transformed_axis_dir)
    b_axis = np.array(vector_norm(np.cross(transformed_axis_dir, d)))
    #print("b", b_axis)
    
    # compute parameters
    original_center = [x, y, 0., 1.]
    transformed_center = np.matmul(trans_mat, np.array(original_center))[:-1]
    r_axis = math.sqrt(x**2 + y**2)

    # sample points in rings along axis
    no_of_axis_points = 100    
    no_of_ring_points = 20
    ring_points = []

    # iterate through rings
    for i in range(no_of_axis_points):
        
        #generate a point on the elbow axis
        angle_axis = (a/no_of_axis_points)*i 
        axis_point = (transformed_center + (r_axis * math.cos(angle_axis) * b_axis) 
                        - (r_axis * math.sin(angle_axis) * np.array(d)))
     
        # find axes of ring plane
        if (i == 0):
            ring_x_init = vector_norm(axis_point - transformed_center)
            ring_y = vector_norm(np.cross(d, ring_x_init))
        ring_x = vector_norm(axis_point - transformed_center)
                
        # iterate through points in each ring
        for j in range(no_of_ring_points):

            # generate random point on ring around axis point
            angle_ring = random.uniform(0., 2*math.pi)
            ring_point = (axis_point + r*math.cos(angle_ring)*np.array(ring_x) 
                            - r*math.sin(angle_ring)*np.array(ring_y))
            ring_points.append(ring_point)

#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(ring_points)

    return ring_points



# recover axis direction from six trig values starting from index k
def get_direction_from_trig_tensor(preds_tensor, k):
    return(torch.transpose(torch.vstack((torch.atan2(preds_tensor[:,k], preds_tensor[:,k+1]),
                                         torch.atan2(preds_tensor[:,k+2], preds_tensor[:,k+3]),
                                         torch.atan2(preds_tensor[:,k+4], preds_tensor[:,k+5]))),
                           0, 1))


# generate points on surface of elbow
def generate_elbow_cloud_tensor(preds_tensor):
    # t1 = time.perf_counter()
    # read params
    tensor_size = preds_tensor.shape[0]
    r, x, y = preds_tensor[:,0], preds_tensor[:,1], preds_tensor[:,2]
    d = get_direction_from_trig_tensor(preds_tensor, 8)
    a = torch.atan2(preds_tensor[:,3], preds_tensor[:,4])
    p = torch.transpose(torch.vstack((preds_tensor[:,5], 
                                      preds_tensor[:,6], 
                                      preds_tensor[:,7])), 
                        0, 1)

    # get new coordinate frame of elbow
    old_z = torch.tensor((0., 0., 1.))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # compute transformation
    tensor_0 = torch.zeros(tensor_size).cuda()
    tensor_1 = torch.ones(tensor_size).cuda()
    trans_mat = torch.transpose(torch.transpose(torch.stack((torch.column_stack((x_axis, tensor_0)), 
                                                           torch.column_stack((y_axis, tensor_0)), 
                                                           torch.column_stack((d, tensor_0)),
                                                           torch.column_stack((p, tensor_1)))),
                                              0, 1),
                              1,2)

    # compute axis
    theta = torch.atan2(x, y)
    original_axis_dir = torch.transpose(torch.vstack((torch.cos(theta), 
                                                      -1*torch.sin(theta), 
                                                      tensor_0,
                                                      tensor_0)), 
                                        0, 1)
    original_axis_dir = original_axis_dir[:, :, None]
    transformed_axis_dir = torch.flatten(torch.bmm(trans_mat, 
                                                   original_axis_dir), 
                                         start_dim=1)[:, :-1]
    b_axis = F.normalize(torch.cross(transformed_axis_dir, d))

    # compute parameters
    original_center =  torch.transpose(torch.vstack((x, 
                                                     y, 
                                                     tensor_0,
                                                     tensor_1)),
                                       0, 1)
    original_center = original_center[:, :, None]
    transformed_center = torch.flatten(torch.bmm(trans_mat, 
                                                 original_center), 
                                       start_dim=1)[:, :-1]
    r_axis = torch.sqrt(torch.square(x) + torch.square(y))

    # sample points in rings along axis
    no_of_axis_points = 100
    no_of_ring_points = 20
    ring_points = torch.zeros((tensor_size, no_of_axis_points*no_of_ring_points, 3)).cuda()
    count = 0

    # t2 = time.perf_counter()

    # iterate through rings
    for i in range(no_of_axis_points):
        
        # generate a point on the elbow axis
        angle_axis = (a/no_of_axis_points)*i
        axis_point = (transformed_center + 
                      ((r_axis * torch.cos(angle_axis)).unsqueeze(1) * b_axis) - 
                      ((r_axis * torch.sin(angle_axis)).unsqueeze(1) * d))
     
        # find axes of ring plane
        if (i == 0):
            ring_x_init = F.normalize(axis_point - transformed_center)
            ring_y = F.normalize(torch.cross(d, ring_x_init))
        ring_x = F.normalize(axis_point - transformed_center)
                
        # iterate through points in each ring
        for j in range(no_of_ring_points):

            # generate random point on ring around axis point
            #angle_ring = torch.rand(tensor_size).cuda()*2*math.pi
            angle_ring = torch.tensor(j*2*math.pi/no_of_ring_points)
            ring_point = (axis_point + 
                          ((r * torch.cos(angle_ring)).unsqueeze(1) * ring_x) - 
                          ((r * torch.sin(angle_ring)).unsqueeze(1) * ring_y))
            ring_points[:, count] = ring_point

            count += 1

    # t3 = time.perf_counter()
    # print("cloud", t2-t1, "chamf", t3-t2)

    return(ring_points)



