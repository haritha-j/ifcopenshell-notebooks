import math
import random
import time
import torch
import numpy as np
from chamferdist import ChamferDistance
import torch.nn.functional as F

from src.visualisation import get_direction_from_trig
from src.geometry import vector_normalise, vector_mag


# generate points on surface of elbow
def generate_elbow_cloud(preds):
    # read params
    r, x, y = preds[0], preds[1], preds[2], 
    d = get_direction_from_trig(preds, 8)
    a = math.atan2(preds[6], preds[7])
    p = [preds[3], preds[4], preds[5]]
    
    # get new coordinate frame of elbow
    old_z = (0., 0., 1.)
    x_axis = vector_normalise(np.cross(d, old_z))
    y_axis = vector_normalise(np.cross(d, x_axis))
    
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
    b_axis = np.array(vector_normalise(np.cross(transformed_axis_dir, d)))
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
            ring_x_init = vector_normalise(axis_point - transformed_center)
            ring_y = vector_normalise(np.cross(d, ring_x_init))
        ring_x = vector_normalise(axis_point - transformed_center)
                
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

# sample points in rings along axis of a cylinder
def get_cylinder_points(no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis):
    ring_points = []

    # iterate through rings
    for i in range(no_of_axis_points):
        
        #generate a point on the elbow axis
        delta_axis = (l/no_of_axis_points)*i
        axis_point = [p[i] + d[i]*delta_axis for i in range(3)]
                
        # iterate through points in each ring
        for j in range(no_of_ring_points):

            # generate random point on ring around axis point
            angle_ring = random.uniform(0., 2*math.pi)
            ring_point = (axis_point + 
                          r*math.cos(angle_ring)*np.array(x_axis) - 
                          r*math.sin(angle_ring)*np.array(y_axis))
            ring_points.append(ring_point)

    return ring_points


# generate points on surface of pipe
def generate_pipe_cloud(preds):
    # read params
    r, l = preds[0], preds[1]
    d = get_direction_from_trig(preds, 5)
    p0 = [preds[2], preds[3], preds[4]]
    p = [p0[i] - ((l*d[i])/2) for i in range(3)]
    
    # get new coordinate frame of pipe
    old_z = (0., 0., 1.)
    x_axis = vector_normalise(np.cross(d, old_z))
    y_axis = vector_normalise(np.cross(d, x_axis))
    
    # sample points in rings along axis
    no_of_axis_points = 50    
    no_of_ring_points = 40
    ring_points = get_cylinder_points(no_of_axis_points, no_of_ring_points, 
                                      r, l, p, d, x_axis, y_axis)

    return ring_points


# generate points on surface of tee
def generate_tee_cloud(preds):
    # read params
    r1, l1, r2, l2 = preds[0], preds[1], preds[2], preds[3]
    d1 = get_direction_from_trig(preds, 7)
    d2 = get_direction_from_trig(preds, 13)
    p2 = [preds[4], preds[5], preds[6]]
    p1 = [p2[i] - ((l1*d1[i])/2) for i in range(3)]
    
    # get new coordinate frame of tee
    old_z = (0., 0., 1.)
    x_axis = vector_normalise(np.cross(d1, old_z))
    y_axis = vector_normalise(np.cross(d1, x_axis))
    
    # sample points on main tube
    no_of_axis_points = 50    
    no_of_ring_points = 40
    tube1_points = get_cylinder_points(no_of_axis_points, no_of_ring_points, 
                                      r1, l1, p1, d1, x_axis, y_axis)

    # sample points on secondary tube
    x_axis = vector_normalise(np.cross(d2, old_z))
    y_axis = vector_normalise(np.cross(d2, x_axis))   
    tube2_points = get_cylinder_points(no_of_axis_points, no_of_ring_points, 
                                      r2, l2, p2, d2, x_axis, y_axis)
    
    # remove points from secondary tube in main tube
    tube2_points = np.array(tube2_points)
    p1, p2 = np.array(p1), np.array(p2)
    p2p1 = p2-p1
    p2p1_mag = vector_mag(p2p1)
    tube2_points_ref = []
    
    for q in tube2_points:
        dist = vector_mag(np.cross((q-p1), (p2p1))) / p2p1_mag
        #print(dist)
        if dist > r1:
            tube2_points_ref.append(q.tolist())
            
    # remove points from main tube in secondary tube
    tube1_points = np.array(tube1_points)
    p3 = np.array(p2 + d2)
    p2p3 = p2-p3
    p2p3_mag = vector_mag(p2p3)
    tube1_points_ref = []
    
    for q in tube1_points:
        dist = vector_mag(np.cross((q-p3), (p2p3))) / p2p3_mag
        cos_theta = np.dot(q-p2, p2p3)
        if dist > r2 or cos_theta > 0:
            tube1_points_ref.append(q.tolist())
    
    # make sure not all points are deleted if predictions are very wrong
    thresh = 50
    if len(tube1_points_ref) < thresh and len(tube2_points_ref) < thresh:
        return (tube1_points.tolist() + tube2_points.tolist())
    elif len(tube2_points_ref) < thresh:
        return (tube1_points_ref + tube2_points.tolist())
    elif len(tube1_points_ref) < thresh:
        return (tube1_points.tolist() + tube2_points_ref)
    else:
        return (tube1_points_ref + tube2_points_ref)


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
    a = torch.atan2(preds_tensor[:,6], preds_tensor[:,7])
    p = torch.transpose(torch.vstack((preds_tensor[:,3], 
                                      preds_tensor[:,4], 
                                      preds_tensor[:,5])), 
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


# sample points in rings along axis of a cylinder
def get_cylinder_points_tensor(no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis, tensor_size):
    count = 0
    ring_points = torch.zeros((tensor_size, no_of_axis_points*no_of_ring_points, 3)).cuda()

    # iterate through rings
    for i in range(no_of_axis_points):
        
        # generate a point on the elbow axis
        delta_axis = (l/no_of_axis_points)*i
        axis_point = p + d * torch.unsqueeze(delta_axis, 1)
                
        # iterate through points in each ring
        for j in range(no_of_ring_points):

            # generate random point on ring around axis point
            #angle_ring = torch.rand(tensor_size).cuda()*2*math.pi
            angle_ring = torch.tensor(j*2*math.pi/no_of_ring_points)
            ring_point = (axis_point + 
                          ((r * torch.cos(angle_ring)).unsqueeze(1) * x_axis) - 
                          ((r * torch.sin(angle_ring)).unsqueeze(1) * y_axis))
            ring_points[:, count] = ring_point

            count += 1

    return(ring_points)


# generate points on surface of cylinder
def generate_pipe_cloud_tensor(preds_tensor):
    # read params
    tensor_size = preds_tensor.shape[0]
    r, l = preds_tensor[:,0], preds_tensor[:,1]
    d = get_direction_from_trig_tensor(preds_tensor, 5)
    p0 = torch.transpose(torch.vstack((preds_tensor[:,2], 
                                      preds_tensor[:,3], 
                                      preds_tensor[:,4])), 
                        0, 1)
    p = p0 - (d * l[:, None]/2)

    # get new coordinate frame of pipe
    old_z = torch.tensor((0., 0., 1.))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # sample points in rings along axis
    # TODO: dynamically balance ring and axis points
    no_of_axis_points = 50
    no_of_ring_points = 40
    ring_points = get_cylinder_points_tensor(no_of_axis_points, no_of_ring_points, r,
                                             l, p, d, x_axis, y_axis, tensor_size)

    return(ring_points)


# generate points on surface of tee
def generate_tee_cloud_tensor(preds_tensor):
    # read params
    tensor_size = preds_tensor.shape[0]
    r1, l1, r2, l2 = preds_tensor[:,0], preds_tensor[:,1], preds_tensor[:,2], preds_tensor[:,3]
    d1 = get_direction_from_trig_tensor(preds_tensor, 7)
    d2 = get_direction_from_trig_tensor(preds_tensor, 13)
    p2 = torch.transpose(torch.vstack((preds_tensor[:,4], 
                                      preds_tensor[:,5], 
                                      preds_tensor[:,6])), 
                        0, 1)
    p1 = p2 - (d1 * l1[:, None]/2)

    # get new coordinate frame of pipe
    old_z = torch.tensor((0., 0., 1.))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d1, old_z))
    y_axis = F.normalize(torch.cross(d1, x_axis))

    # sample points in rings along main tube
    no_of_axis_points = 50
    no_of_ring_points = 40
    tube1_points = get_cylinder_points_tensor(no_of_axis_points, no_of_ring_points, r1, 
                                             l1, p1, d1, x_axis, y_axis, tensor_size)
    
    # sample points on secondary tube
    x_axis = F.normalize(torch.cross(d2, old_z))
    y_axis = F.normalize(torch.cross(d2, x_axis))   
    tube2_points = get_cylinder_points_tensor(no_of_axis_points, no_of_ring_points, r2,
                                              l2, p2, d2, x_axis, y_axis, tensor_size)
    
    # remove points from secondary tube in main tube
    thresh = 50
    p2p1 = p2-p1
    p2p1_mag = torch.linalg.vector_norm(p2p1, dim=1)
    tube2_error = False
    no_points = tube2_points.shape[1]
    tube2_points_ref = torch.zeros(tube2_points.shape).cuda()
    tube2_points_mask = torch.zeros((tensor_size, no_of_axis_points*no_of_ring_points), 
                                    dtype=torch.bool).cuda()
    
    for i in range(tube2_points.shape[1]):
        cr = torch.cross(tube2_points[:,i]-p1, p2p1)
        dist = torch.linalg.vector_norm(cr, dim=1) / p2p1_mag
        tube2_points_mask[:, i] = torch.le(r1, dist)

    for i in range(len(tube2_points)):
        cl = tube2_points[i][tube2_points_mask[i]]
        if (len(cl) < thresh):
            tube2_error = True
            break
        tensor_repeat = cl[0]
        pad = tensor_repeat.unsqueeze(0).repeat(no_points-cl.shape[0], 1)
        cl = torch.cat((cl, pad), 0)
        tube2_points_ref[i] = cl
    
    # remove points from main tube in secondary tube
    p3 = p2+d2
    p2p3 = p2-p3
    p2p3_mag = torch.linalg.vector_norm(p2p3, dim=1)
    tube1_error = False
    no_points = tube1_points.shape[1]
    tube1_points_ref = torch.zeros(tube1_points.shape).cuda()
    tube1_points_mask = torch.zeros((tensor_size, no_of_axis_points*no_of_ring_points), 
                                    dtype=torch.bool).cuda()
   
    for i in range(tube1_points.shape[1]):
        cr = torch.cross(tube1_points[:,i]-p3, p2p3)
        dist = torch.linalg.vector_norm(cr, dim=1) / p2p3_mag
        cos_theta = torch.sum((tube1_points[:,i]-p2) * p2p3, dim =-1)
        tube1_points_mask[:, i] = torch.logical_or(torch.le(r2, dist), torch.ge(cos_theta, 0))

    for i in range(len(tube1_points)):
        cl = tube1_points[i][tube1_points_mask[i]]
        if (len(cl) < thresh):
            tube1_error = True
            break
        tensor_repeat = cl[0]
        pad = tensor_repeat.unsqueeze(0).repeat(no_points-cl.shape[0], 1)
        cl = torch.cat((cl, pad), 0)
        tube1_points_ref[i] = cl       
        
    if tube1_error and tube2_error:            
        return torch.cat((tube1_points, tube2_points), 1)
    elif tube1_error:
        return torch.cat((tube1_points, tube2_points_ref), 1)
    elif tube2_error:
         return torch.cat((tube1_points_ref, tube2_points), 1)
    else:
        return torch.cat((tube1_points_ref, tube2_points_ref), 1)    


def get_chamfer_dist_single(src, preds, cat):
    if cat == "elbow":
        tgt = generate_elbow_cloud(preds)
    elif cat == "pipe":
        tgt = generate_pipe_cloud(preds)
    elif cat == "tee":
        tgt = generate_tee_cloud(preds)
    src, tgt = torch.tensor(np.expand_dims(src, axis=0)).cuda().float(), torch.tensor(np.expand_dims(tgt, axis=0)).cuda().float()
    
    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(src, tgt, bidirectional=True)
    return bidirectional_dist.detach().cpu().item(), tgt


def get_chamfer_loss(preds_tensor, src_pcd_tensor, cat):
    target_pcd_list = []
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    preds_list = preds_tensor.cpu().detach().numpy()
    #t1 = time.perf_counter()
    for preds in preds_list:
        if cat == "elbow":
            target_pcd_list.append(generate_elbow_cloud(preds))
        elif cat == "pipe":
            target_pcd_list.append(generate_pipe_cloud(preds))
        elif cat == "tee":
            target_pcd_list.append(generate_tee_cloud(preds))

    target_pcd_tensor = torch.tensor(target_pcd_list).float().cuda()
    #t2 = time.perf_counter()

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(target_pcd_tensor, src_pcd_tensor, bidirectional=True)
    #t3 = time.perf_counter()
    #print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


def get_chamfer_loss_tensor(preds_tensor, src_pcd_tensor, cat):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    
    if cat == "elbow":
        target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
    elif cat == "pipe":
        target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
    elif cat == "tee":
        target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor)
    #t2 = time.perf_counter()
    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(target_pcd_tensor, src_pcd_tensor, bidirectional=True)
    #t3 = time.perf_counter()
    #print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


