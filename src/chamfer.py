import math
import random
import time
import torch
import numpy as np
from chamferdist import ChamferDistance
import torch.nn.functional as F

from src.visualisation import get_direction_from_trig
from src.geometry import *
from utils.EMD import emd_module as emd


# calculate approximate earth mover's distance
# NOTE: gradient is only calculated for output, not gt
def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, assignment = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sum(dist)/len(output)*2
    return emd_out, assignment


# generate points on surface of elbow
def generate_elbow_cloud(preds, return_elbow_edge=False):
    # read params
    r, x, y = (
        preds[0],
        preds[1],
        preds[2],
    )
    d = vector_normalise(get_direction_from_trig(preds, 8))
    a = math.atan2(preds[6], preds[7])
    p = [preds[3], preds[4], preds[5]]

    # get new coordinate frame of elbow
    old_z = (0.0, 0.0, 1.0)
    if np.isclose(np.dot(d, old_z), 1) or np.isclose(np.dot(d, old_z), -1):
        old_z = (0.0, 1.0, 0.0)

    x_axis = vector_normalise(np.cross(d, old_z))
    y_axis = vector_normalise(np.cross(d, x_axis))

    # compute transformation
    rot_mat = np.transpose(np.array([x_axis, y_axis, d]))
    t = np.array([[p[0]], [p[1]], [p[2]]])

    trans_mat = np.vstack((np.hstack((rot_mat, t)), [0.0, 0.0, 0.0, 1.0]))

    # compute axis
    theta = math.atan2(x, y)
    original_axis_dir = [math.cos(theta), -1 * math.sin(theta), 0.0, 0.0]
    transformed_axis_dir = np.matmul(trans_mat, np.array(original_axis_dir))[:-1]
    # print(transformed_axis_dir)
    b_axis = np.array(vector_normalise(np.cross(transformed_axis_dir, d)))
    # print("b", b_axis)

    # compute parameters
    original_center = [x, y, 0.0, 1.0]
    transformed_center = np.matmul(trans_mat, np.array(original_center))[:-1]
    r_axis = math.sqrt(x**2 + y**2)

    # sample points in rings along axis
    no_of_axis_points = 100
    no_of_ring_points = 20
    ring_points = []

    if return_elbow_edge:
        delta = 0.01
        return (
            (
                transformed_center
                + (r_axis * math.cos(a) * b_axis)
                - (r_axis * math.sin(a) * np.array(d))
            ),
            (
                transformed_center
                + (r_axis * math.cos(a - delta) * b_axis)
                - (r_axis * math.sin(a - delta) * np.array(d))
            ),
            (
                transformed_center
                + (r_axis * math.cos(a + delta) * b_axis)
                - (r_axis * math.sin(a + delta) * np.array(d))
            ),
        )

    # iterate through rings
    for i in range(no_of_axis_points):
        # generate a point on the elbow axis
        angle_axis = (a / no_of_axis_points) * i
        axis_point = (
            transformed_center
            + (r_axis * math.cos(angle_axis) * b_axis)
            - (r_axis * math.sin(angle_axis) * np.array(d))
        )

        # find axes of ring plane
        if i == 0:
            ring_x_init = vector_normalise(axis_point - transformed_center)
            ring_y = vector_normalise(np.cross(d, ring_x_init))
        ring_x = vector_normalise(axis_point - transformed_center)

        # iterate through points in each ring
        for j in range(no_of_ring_points):
            # generate random point on ring around axis point
            angle_ring = random.uniform(0.0, 2 * math.pi)
            ring_point = (
                axis_point
                + r * math.cos(angle_ring) * np.array(ring_x)
                - r * math.sin(angle_ring) * np.array(ring_y)
            )
            ring_points.append(ring_point)

    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(ring_points)

    return ring_points


# sample points in rings along axis of a cylinder
def get_cylinder_points(
    no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis
):
    ring_points = []

    # iterate through rings
    for i in range(no_of_axis_points):
        # generate a point on the elbow axis
        delta_axis = (l / no_of_axis_points) * i
        axis_point = [p[i] + d[i] * delta_axis for i in range(3)]

        # iterate through points in each ring
        for j in range(no_of_ring_points):
            # generate random point on ring around axis point
            angle_ring = random.uniform(0.0, 2 * math.pi)
            ring_point = (
                axis_point
                + r * math.cos(angle_ring) * np.array(x_axis)
                - r * math.sin(angle_ring) * np.array(y_axis)
            )
            ring_points.append(ring_point)

    return ring_points


# generate points on surface of flange
def generate_flange_cloud(preds, disc=True):
    # read params
    r1, r2, l1, l2 = preds[0], preds[1], preds[2], preds[3]
    d = vector_normalise(get_direction_from_trig(preds, 7))
    # d = [preds[7], preds[8], preds[9]]
    p0 = [preds[3], preds[5], preds[6]]
    p = [p0[i] - ((l1 * d[i])) for i in range(3)]
    p1 = [p0[i] + ((l2 * d[i])) for i in range(3)]

    # get new coordinate frame of flange
    old_z = (0.0, 0.0, 1.0)
    if np.isclose(np.dot(d, old_z), 1) or np.isclose(np.dot(d, old_z), -1):
        old_z = (0.0, 1.0, 0.0)

    x_axis = vector_normalise(np.cross(d, old_z))
    y_axis = vector_normalise(np.cross(d, x_axis))

    # sample points in rings along axis
    if disc:
        no_of_axis_points = 5
        no_of_ring_points = 50
    else:
        no_of_axis_points = 5
        no_of_ring_points = 100
    ring_points1 = get_cylinder_points(
        no_of_axis_points, no_of_ring_points, r1, l1, p, d, x_axis, y_axis
    )
    ring_points2 = get_cylinder_points(
        no_of_axis_points, no_of_ring_points, r2, l2, p0, d, x_axis, y_axis
    )

    # sample points on discs on the ends of flange
    if disc:
        disc_points = []
        for i in range(1, 6):
            disc_points += get_cylinder_points(
                1, no_of_ring_points, i * r2 / 6, l1, p1, d, x_axis, y_axis
            )
        for i in range(3, 6):
            disc_points += get_cylinder_points(
                1, no_of_ring_points, i * r2 / 6, l1, p0, d, x_axis, y_axis
            )
        for i in range(1, 3):
            disc_points += get_cylinder_points(
                1, no_of_ring_points, i * r2 / 6, l1, p, d, x_axis, y_axis
            )
        return ring_points1 + ring_points2 + disc_points

    else:
        return ring_points1 + ring_points2


# generate points on surface of pipe
def generate_pipe_cloud(preds, scale=False):
    # read params
    r, l = preds[0], preds[1]
    d = vector_normalise(get_direction_from_trig(preds, 5))
    p0 = [preds[2], preds[3], preds[4]]
    p = [p0[i] - ((l * d[i]) / 2) for i in range(3)]

    # get new coordinate frame of pipe
    old_z = (0.0, 0.0, 1.0)
    if np.isclose(np.dot(d, old_z), 1) or np.isclose(np.dot(d, old_z), -1):
        old_z = (0.0, 1.0, 0.0)

    x_axis = vector_normalise(np.cross(d, old_z))
    y_axis = vector_normalise(np.cross(d, x_axis))

    # sample points in rings along axis
    if scale:
        no_of_axis_points = int(50 * l) if l > 1.0 else 50
    else:
        no_of_axis_points = 50
    no_of_ring_points = 40
    ring_points = get_cylinder_points(
        no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis
    )

    return ring_points


# generate points on surface of tee
def generate_tee_cloud(preds, refine=True):
    # read params
    r1, l1, r2, l2 = preds[0], preds[1], preds[2], preds[3]
    d1 = vector_normalise(get_direction_from_trig(preds, 7))
    d2 = vector_normalise(get_direction_from_trig(preds, 13))
    p2 = [preds[4], preds[5], preds[6]]
    p1 = [p2[i] - ((l1 * d1[i]) / 2) for i in range(3)]

    # get new coordinate frame of tee
    old_z = (0.0, 0.0, 1.0)
    if np.isclose(np.dot(d1, old_z), 1) or np.isclose(np.dot(d1, old_z), -1):
        old_z = (0.0, 1.0, 0.0)
    x_axis = vector_normalise(np.cross(d1, old_z))
    y_axis = vector_normalise(np.cross(d1, x_axis))

    # sample points on main tube
    no_of_axis_points = 50
    no_of_ring_points = 40
    tube1_points = get_cylinder_points(
        no_of_axis_points, no_of_ring_points, r1, l1, p1, d1, x_axis, y_axis
    )

    # sample points on secondary tube
    x_axis = vector_normalise(np.cross(d2, old_z))
    y_axis = vector_normalise(np.cross(d2, x_axis))
    tube2_points = get_cylinder_points(
        no_of_axis_points, no_of_ring_points, r2, l2, p2, d2, x_axis, y_axis
    )

    if not refine:
        return tube1_points + tube2_points
    else:
        # remove points from secondary tube in main tube
        tube2_points = np.array(tube2_points)
        p1, p2 = np.array(p1), np.array(p2)
        p2p1 = p2 - p1
        p2p1_mag = vector_mag(p2p1)
        tube2_points_ref = []

        for q in tube2_points:
            dist = vector_mag(np.cross((q - p1), (p2p1))) / p2p1_mag
            # print(dist)
            if dist > r1:
                tube2_points_ref.append(q.tolist())

        # remove points from main tube in secondary tube
        tube1_points = np.array(tube1_points)
        p3 = np.array(p2 + d2)
        p2p3 = p2 - p3
        p2p3_mag = vector_mag(p2p3)
        tube1_points_ref = []

        for q in tube1_points:
            dist = vector_mag(np.cross((q - p3), (p2p3))) / p2p3_mag
            cos_theta = np.dot(q - p2, p2p3)
            if dist > r2 or cos_theta > 0:
                tube1_points_ref.append(q.tolist())

        # make sure not all points are deleted if predictions are very wrong
        thresh = 50
        if len(tube1_points_ref) < thresh and len(tube2_points_ref) < thresh:
            return tube1_points.tolist() + tube2_points.tolist()
        elif len(tube2_points_ref) < thresh:
            return tube1_points_ref + tube2_points.tolist()
        elif len(tube1_points_ref) < thresh:
            return tube1_points.tolist() + tube2_points_ref
        else:
            return tube1_points_ref + tube2_points_ref


# recover axis direction from six trig values starting from index k
def get_direction_from_trig_tensor(preds_tensor, k):
    return torch.transpose(
        torch.vstack(
            (
                torch.atan2(preds_tensor[:, k], preds_tensor[:, k + 1]),
                torch.atan2(preds_tensor[:, k + 2], preds_tensor[:, k + 3]),
                torch.atan2(preds_tensor[:, k + 4], preds_tensor[:, k + 5]),
            )
        ),
        0,
        1,
    )


# sample points in rings along axis of a cylinder
def get_cylinder_points_tensor(
    no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis, tensor_size
):
    count = 0
    ring_points = torch.zeros(
        (tensor_size, no_of_axis_points * no_of_ring_points, 3)
    ).cuda()

    # iterate through rings
    for i in range(no_of_axis_points):
        # generate a point on the elbow axis
        delta_axis = (l / no_of_axis_points) * i
        axis_point = p + d * torch.unsqueeze(delta_axis, 1)

        # iterate through points in each ring
        for j in range(no_of_ring_points):
            # generate random point on ring around axis point
            # angle_ring = torch.rand(tensor_size).cuda()*2*math.pi
            angle_ring = torch.tensor(j * 2 * math.pi / no_of_ring_points)
            ring_point = (
                axis_point
                + ((r * torch.cos(angle_ring)).unsqueeze(1) * x_axis)
                - ((r * torch.sin(angle_ring)).unsqueeze(1) * y_axis)
            )
            ring_points[:, count] = ring_point

            count += 1

    return ring_points


# sample points in rings on circular surface
def get_circle_points_tensor(
    no_of_rings, no_of_ring_points, r, p, x_axis, y_axis, tensor_size
):
    count = 0
    no_of_points = no_of_ring_points * sum([i for i in range(1, no_of_rings + 1)])
    ring_points = torch.zeros((tensor_size, no_of_points, 3)).cuda()

    # iterate through rings
    for i in range(1, no_of_rings + 1):
        # generate a point on the elbow axis
        delta_r = (r / no_of_rings) * i

        # iterate through points in each ring
        for j in range(no_of_ring_points * i):
            # generate random point on ring around axis point
            # angle_ring = torch.rand(tensor_size).cuda()*2*math.pi
            angle_ring = torch.tensor(j * 2 * math.pi / (no_of_ring_points * i))
            ring_point = (
                p
                + ((delta_r * torch.cos(angle_ring)).unsqueeze(1) * x_axis)
                - ((delta_r * torch.sin(angle_ring)).unsqueeze(1) * y_axis)
            )
            ring_points[:, count] = ring_point

            count += 1

    return ring_points


# generate points on surface of flange
def generate_flange_cloud_tensor(preds_tensor, disc=True):
    # read params
    tensor_size = preds_tensor.shape[0]
    r1, r2, l1, l2 = (
        preds_tensor[:, 0],
        preds_tensor[:, 1],
        preds_tensor[:, 2],
        preds_tensor[:, 3],
    )
    d = F.normalize(get_direction_from_trig_tensor(preds_tensor, 7))
    p0 = torch.transpose(
        torch.vstack((preds_tensor[:, 4], preds_tensor[:, 5], preds_tensor[:, 6])), 0, 1
    )
    p = p0 - (d * l1[:, None])
    p1 = p0 + (d * l2[:, None])

    # get new coordinate frame of pipe
    old_z = torch.tensor((0.0, 0.0, 1.0))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # sample points in rings along axis
    if disc:
        no_of_axis_points = 8
        no_of_ring_points = 68
    else:
        no_of_axis_points = 8
        no_of_ring_points = 128

    ring_points1 = get_cylinder_points_tensor(
        no_of_axis_points, no_of_ring_points, r1, l1, p, d, x_axis, y_axis, tensor_size
    )
    ring_points2 = get_cylinder_points_tensor(
        no_of_axis_points, no_of_ring_points, r2, l2, p0, d, x_axis, y_axis, tensor_size
    )

    if disc:
        no_of_ring_points = 96
        disc_points = torch.zeros((tensor_size, no_of_ring_points * 10, 3)).cuda()
        for i in range(1, 6):
            disc_points[
                :, no_of_ring_points * (i - 1) : no_of_ring_points * i
            ] = get_cylinder_points_tensor(
                1, no_of_ring_points, i * r2 / 6, l1, p1, d, x_axis, y_axis, tensor_size
            )
        for i in range(3, 6):
            disc_points[
                :, no_of_ring_points * (i + 2) : no_of_ring_points * (i + 3)
            ] = get_cylinder_points_tensor(
                1, no_of_ring_points, i * r2 / 6, l1, p0, d, x_axis, y_axis, tensor_size
            )
        for i in range(1, 3):
            disc_points[
                :, no_of_ring_points * (i + 7) : no_of_ring_points * (i + 8)
            ] = get_cylinder_points_tensor(
                1, no_of_ring_points, i * r2 / 6, l1, p, d, x_axis, y_axis, tensor_size
            )
        return torch.cat((ring_points1, ring_points2, disc_points), 1)

    else:
        return torch.cat((ring_points1, ring_points2), 1)


# generate points on surface of cylinder
def generate_pipe_cloud_tensor(preds_tensor):
    # read params
    tensor_size = preds_tensor.shape[0]
    r, l = preds_tensor[:, 0], preds_tensor[:, 1]
    d = F.normalize(get_direction_from_trig_tensor(preds_tensor, 5))
    p0 = torch.transpose(
        torch.vstack((preds_tensor[:, 2], preds_tensor[:, 3], preds_tensor[:, 4])), 0, 1
    )
    p = p0 - (d * l[:, None] / 2)

    # get new coordinate frame of pipe
    old_z = torch.tensor((0.0, 0.0, 1.0))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # sample points in rings along axis
    # TODO: dynamically balance ring and axis points
    no_of_axis_points = 32
    no_of_ring_points = 32
    ring_points = get_cylinder_points_tensor(
        no_of_axis_points, no_of_ring_points, r, l, p, d, x_axis, y_axis, tensor_size
    )

    return ring_points


# generate points on surface of elbow
def generate_elbow_cloud_tensor(preds_tensor, return_elbow_edge=False):
    # t1 = time.perf_counter()
    # read params
    tensor_size = preds_tensor.shape[0]
    r, x, y = preds_tensor[:, 0], preds_tensor[:, 1], preds_tensor[:, 2]
    d = F.normalize(get_direction_from_trig_tensor(preds_tensor, 8))
    a = torch.atan2(preds_tensor[:, 6], preds_tensor[:, 7])
    p = torch.transpose(
        torch.vstack((preds_tensor[:, 3], preds_tensor[:, 4], preds_tensor[:, 5])), 0, 1
    )

    # get new coordinate frame of elbow
    old_z = torch.tensor((0.0, 0.0, 1.0))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # compute transformation
    tensor_0 = torch.zeros(tensor_size).cuda()
    tensor_1 = torch.ones(tensor_size).cuda()
    trans_mat = torch.transpose(
        torch.transpose(
            torch.stack(
                (
                    torch.column_stack((x_axis, tensor_0)),
                    torch.column_stack((y_axis, tensor_0)),
                    torch.column_stack((d, tensor_0)),
                    torch.column_stack((p, tensor_1)),
                )
            ),
            0,
            1,
        ),
        1,
        2,
    )

    # compute axis
    theta = torch.atan2(x, y)
    original_axis_dir = torch.transpose(
        torch.vstack((torch.cos(theta), -1 * torch.sin(theta), tensor_0, tensor_0)),
        0,
        1,
    )
    original_axis_dir = original_axis_dir[:, :, None]
    transformed_axis_dir = torch.flatten(
        torch.bmm(trans_mat, original_axis_dir), start_dim=1
    )[:, :-1]
    b_axis = F.normalize(torch.cross(transformed_axis_dir, d))

    # compute parameters
    original_center = torch.transpose(torch.vstack((x, y, tensor_0, tensor_1)), 0, 1)
    original_center = original_center[:, :, None]
    transformed_center = torch.flatten(
        torch.bmm(trans_mat, original_center), start_dim=1
    )[:, :-1]
    r_axis = torch.sqrt(torch.square(x) + torch.square(y))

    # sample points in rings along axis
    no_of_axis_points = 128
    no_of_ring_points = 16
    # ring_points = torch.zeros(1,1,1).cuda().expand(tensor_size, no_of_axis_points*no_of_ring_points, 3)
    # ring_points = torch.zeros((tensor_size, no_of_axis_points*no_of_ring_points, 3)).cuda()
    ring_points_list = []
    count = 0

    if return_elbow_edge:
        delta = 0.01
        return (
            transformed_center
            + ((r_axis * torch.cos(a)).unsqueeze(1) * b_axis)
            - ((r_axis * torch.sin(a)).unsqueeze(1) * d),
            transformed_center
            + ((r_axis * torch.cos(a - delta)).unsqueeze(1) * b_axis)
            - ((r_axis * torch.sin(a - delta)).unsqueeze(1) * d),
            transformed_center
            + ((r_axis * torch.cos(a + delta)).unsqueeze(1) * b_axis)
            - ((r_axis * torch.sin(a + delta)).unsqueeze(1) * d),
        )

    # t2 = time.perf_counter()

    # iterate through rings
    for i in range(no_of_axis_points):
        # generate a point on the elbow axis
        angle_axis = (a / no_of_axis_points) * i
        axis_point = (
            transformed_center
            + ((r_axis * torch.cos(angle_axis)).unsqueeze(1) * b_axis)
            - ((r_axis * torch.sin(angle_axis)).unsqueeze(1) * d)
        )

        # find axes of ring plane
        if i == 0:
            ring_x_init = F.normalize(axis_point - transformed_center)
            ring_y = F.normalize(torch.cross(d, ring_x_init))
        ring_x = F.normalize(axis_point - transformed_center)

        # iterate through points in each ring
        # j = torch.arange(0, no_of_ring_points).cuda()
        # j = j.unsqueeze(0).repeat(tensor_size, 1)
        # print("j shape", j.shape)

        # angle_ring = torch.mul(j,2*math.pi/no_of_ring_points)
        # print("ang ring shape", angle_ring.shape)

        # ring_point = (axis_point +
        #                   (torch.mul(r, torch.cos(angle_ring)).unsqueeze(1) * ring_x) -
        #                   (torch.mul(r, torch.sin(angle_ring)).unsqueeze(1) * ring_y))
        # print("ring_point shape", ring_point.shape)

        # ring_points[:,count:count+no_of_ring_points] = ring_point
        # count += 20
        for j in range(no_of_ring_points):
            # generate random point on ring around axis point
            # angle_ring = torch.rand(tensor_size).cuda()*2*math.pi
            angle_ring = torch.tensor(j * 2 * math.pi / no_of_ring_points)
            ring_point = (
                axis_point
                + ((r * torch.cos(angle_ring)).unsqueeze(1) * ring_x)
                - ((r * torch.sin(angle_ring)).unsqueeze(1) * ring_y)
            )
            # ring_points[:, count] = ring_point
            rp = torch.unsqueeze(ring_point, 1)
            ring_points_list.append(rp)
            count += 1
    ring_points = torch.hstack(ring_points_list)

    # t3 = time.perf_counter()
    # print("cloud", t2-t1, "chamf", t3-t2)
    return ring_points


# generate points on surface of socket elbow
def generate_socket_elbow_cloud_tensor(preds_tensor):
    # read params
    tensor_size = preds_tensor.shape[0]
    r, x, y, l = (
        torch.clone(preds_tensor[:, 0]),
        torch.clone(preds_tensor[:, 1]),
        torch.clone(preds_tensor[:, 2]),
        torch.clone(preds_tensor[:, 14]),
    )
    d = F.normalize(get_direction_from_trig_tensor(preds_tensor, 8))
    a = torch.atan2(preds_tensor[:, 6], preds_tensor[:, 7])
    p = torch.clone(
        torch.transpose(
            torch.vstack((preds_tensor[:, 3], preds_tensor[:, 4], preds_tensor[:, 5])),
            0,
            1,
        )
    )

    # find start of elbow section
    p1 = p - (d * l[:, None])
    elbow_preds_tensor = preds_tensor.clone()
    elbow_preds_tensor[:, 3] = p1[:, 0]
    elbow_preds_tensor[:, 4] = p1[:, 1]
    elbow_preds_tensor[:, 5] = p1[:, 2]

    # slightly reduce radius, and modify x and y
    elbow_preds_tensor_r = elbow_preds_tensor[:, 0] * 0.9
    r_old = torch.sqrt(torch.square(x) + torch.square(y))
    c_a, s_a = torch.cos(a), torch.sin(a)
    scale_factor = torch.div((r_old - r_old * c_a - l * s_a), (1 - c_a) * r_old)

    elbow_preds_tensor_x = torch.mul(elbow_preds_tensor[:, 1], scale_factor)
    elbow_preds_tensor_y = torch.mul(elbow_preds_tensor[:, 2], scale_factor)
    elbow_preds_tensor_new = torch.column_stack(
        (
            elbow_preds_tensor_r,
            elbow_preds_tensor_x,
            elbow_preds_tensor_y,
            elbow_preds_tensor[:, 3:],
        )
    )
    # elbow_preds_tensor[:,2] = torch.mul(elbow_preds_tensor[:,2], scale_factor)
    # elbow_preds_tensor[:,2] = elbow_preds_tensor[:,2] *scale_factor

    p2, p2a, p2b = generate_elbow_cloud_tensor(
        elbow_preds_tensor_new, return_elbow_edge=True
    )
    d2 = F.normalize(p2a - p2b)
    elbow_points = generate_elbow_cloud_tensor(elbow_preds_tensor_new)

    # generate socket points for first socket
    # get new coordinate frame of pipe
    old_z = torch.tensor((0.0, 0.0, 1.0))
    old_z = old_z.repeat(tensor_size, 1).cuda()

    x_axis = F.normalize(torch.cross(d, old_z))
    y_axis = F.normalize(torch.cross(d, x_axis))

    # sample points in rings along axis
    no_of_axis_points = 10
    no_of_ring_points = 40
    ring_points1 = get_cylinder_points_tensor(
        no_of_axis_points,
        no_of_ring_points,
        r,
        l,
        p,
        -1 * d,
        x_axis,
        y_axis,
        tensor_size,
    )

    # generate socket points for second socket
    x_axis = F.normalize(torch.cross(d2, old_z))
    y_axis = F.normalize(torch.cross(d2, x_axis))

    # sample points in rings along axis
    no_of_axis_points = 10
    no_of_ring_points = 40
    ring_points2 = get_cylinder_points_tensor(
        no_of_axis_points,
        no_of_ring_points,
        r,
        l,
        p2,
        -1 * d2,
        x_axis,
        y_axis,
        tensor_size,
    )

    return torch.cat((elbow_points, ring_points1, ring_points2), 1)
    # return torch.cat((ring_points1, ring_points2), 1)


# generate points on surface of tee
# BUG: if even one tee has an error, points in all tees in batch are not deleted
def generate_tee_cloud_tensor(preds_tensor, bp=False):
    # read params
    tensor_size = preds_tensor.shape[0]
    r1, l1, r2, l2 = (
        preds_tensor[:, 0],
        preds_tensor[:, 1],
        preds_tensor[:, 2],
        preds_tensor[:, 3],
    )
    d1 = F.normalize(get_direction_from_trig_tensor(preds_tensor, 7))
    d2 = F.normalize(get_direction_from_trig_tensor(preds_tensor, 13))
    p2 = torch.transpose(
        torch.vstack((preds_tensor[:, 4], preds_tensor[:, 5], preds_tensor[:, 6])), 0, 1
    )
    p1 = p2 - (d1 * l1[:, None] / 2)

    # get new coordinate frame of pipe
    old_z = torch.tensor((0.0, 0.0, 1.0))
    old_z = old_z.repeat(tensor_size, 1).cuda()
    x_axis = F.normalize(torch.cross(d1, old_z))
    y_axis = F.normalize(torch.cross(d1, x_axis))

    # sample points in rings along main tube
    no_of_axis_points = 50
    no_of_ring_points = 40
    no_of_rings = 5
    no_of_circle_ring_points = 7
    tube1_points = get_cylinder_points_tensor(
        no_of_axis_points,
        no_of_ring_points,
        r1,
        l1,
        p1,
        d1,
        x_axis,
        y_axis,
        tensor_size,
    )

    if bp:
        circle1_points = get_circle_points_tensor(
            no_of_rings, no_of_circle_ring_points, r1, p1, x_axis, y_axis, tensor_size
        )
        p3 = p1 + d1 * torch.unsqueeze(l1, 1)
        circle2_points = get_circle_points_tensor(
            no_of_rings, no_of_circle_ring_points, r1, p3, x_axis, y_axis, tensor_size
        )

    # sample points on secondary tube
    x_axis = F.normalize(torch.cross(d2, old_z))
    y_axis = F.normalize(torch.cross(d2, x_axis))
    tube2_points = get_cylinder_points_tensor(
        no_of_axis_points,
        no_of_ring_points,
        r2,
        l2,
        p2,
        d2,
        x_axis,
        y_axis,
        tensor_size,
    )

    if bp:
        p4 = p2 + d2 * torch.unsqueeze(l2, 1)
        circle3_points = get_circle_points_tensor(
            no_of_rings, no_of_circle_ring_points, r2, p4, x_axis, y_axis, tensor_size
        )
    # remove points from secondary tube in main tube
    thresh = 50
    p2p1 = p2 - p1
    p2p1_mag = torch.linalg.vector_norm(p2p1, dim=1)
    tube2_error = False
    no_points = tube2_points.shape[1]
    tube2_points_ref = torch.zeros(tube2_points.shape).cuda()
    tube2_points_mask = torch.zeros(
        (tensor_size, no_of_axis_points * no_of_ring_points), dtype=torch.bool
    ).cuda()

    for i in range(tube2_points.shape[1]):
        cr = torch.cross(tube2_points[:, i] - p1, p2p1)
        dist = torch.linalg.vector_norm(cr, dim=1) / p2p1_mag
        tube2_points_mask[:, i] = torch.le(r1, dist)

    for i in range(len(tube2_points)):
        cl = tube2_points[i][tube2_points_mask[i]]
        if len(cl) < thresh:
            tube2_error = True
            break
        tensor_repeat = cl[0]
        pad = tensor_repeat.unsqueeze(0).repeat(no_points - cl.shape[0], 1)
        cl = torch.cat((cl, pad), 0)
        tube2_points_ref[i] = cl

    # remove points from main tube in secondary tube
    p3 = p2 + d2
    p2p3 = p2 - p3
    p2p3_mag = torch.linalg.vector_norm(p2p3, dim=1)
    tube1_error = False
    no_points = tube1_points.shape[1]
    tube1_points_ref = torch.zeros(tube1_points.shape).cuda()
    tube1_points_mask = torch.zeros(
        (tensor_size, no_of_axis_points * no_of_ring_points), dtype=torch.bool
    ).cuda()

    for i in range(tube1_points.shape[1]):
        cr = torch.cross(tube1_points[:, i] - p3, p2p3)
        dist = torch.linalg.vector_norm(cr, dim=1) / p2p3_mag
        cos_theta = torch.sum((tube1_points[:, i] - p2) * p2p3, dim=-1)
        tube1_points_mask[:, i] = torch.logical_or(
            torch.le(r2, dist), torch.ge(cos_theta, 0)
        )

    for i in range(len(tube1_points)):
        cl = tube1_points[i][tube1_points_mask[i]]
        if len(cl) < thresh:
            tube1_error = True
            break
        tensor_repeat = cl[0]
        pad = tensor_repeat.unsqueeze(0).repeat(no_points - cl.shape[0], 1)
        cl = torch.cat((cl, pad), 0)
        tube1_points_ref[i] = cl

    if tube1_error and tube2_error:
        combined = torch.cat((tube1_points, tube2_points), 1)
    elif tube1_error:
        combined = torch.cat((tube1_points, tube2_points_ref), 1)
    elif tube2_error:
        combined = torch.cat((tube1_points_ref, tube2_points), 1)
    else:
        combined = torch.cat((tube1_points_ref, tube2_points_ref), 1)

    # generate additional points for bp dataset closed tees
    if bp:
        return torch.cat((combined, circle1_points, circle2_points, circle3_points), 1)
    else:
        return combined


def get_chamfer_dist_single(src, preds, cat):
    if cat == "elbow":
        tgt = generate_elbow_cloud(preds)
    elif cat == "pipe":
        tgt = generate_pipe_cloud(preds)
    elif cat == "tee":
        tgt = generate_tee_cloud(preds)
    src, tgt = (
        torch.tensor(np.expand_dims(src, axis=0)).cuda().float(),
        torch.tensor(np.expand_dims(tgt, axis=0)).cuda().float(),
    )

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(src, tgt, bidirectional=True)
    return bidirectional_dist.detach().cpu().item(), tgt


def get_chamfer_loss(preds_tensor, src_pcd_tensor, cat):
    target_pcd_list = []
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    preds_list = preds_tensor.cpu().detach().numpy()
    # t1 = time.perf_counter()
    for preds in preds_list:
        if cat == "elbow":
            target_pcd_list.append(generate_elbow_cloud(preds))
        elif cat == "pipe":
            target_pcd_list.append(generate_pipe_cloud(preds))
        elif cat == "tee":
            target_pcd_list.append(generate_tee_cloud(preds))
        elif cat == "flange":
            target_pcd_list.append(generate_flange_cloud(preds))

    target_pcd_tensor = torch.tensor(target_pcd_list).float().cuda()
    # t2 = time.perf_counter()

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(
        target_pcd_tensor, src_pcd_tensor, bidirectional=True
    )
    # t3 = time.perf_counter()
    # print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


def get_shape_cloud_tensor(preds_tensor, cat):
    if cat == "elbow":
        target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
    elif cat == "pipe":
        target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
    elif cat == "tee":
        target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor, bp=False)
    elif cat == "flange":
        target_pcd_tensor = generate_flange_cloud_tensor(preds_tensor, disc=True)
    elif cat == "socket":
        target_pcd_tensor = generate_socket_elbow_cloud_tensor(preds_tensor)

    return target_pcd_tensor
    

# delta is the constant for robust kernel
# alpha determines the weighting of bidirectional chamfer loss
# this method compares an input point cloud, with cloud generated from predicted params
def get_chamfer_loss_tensor(
    preds_tensor,
    src_pcd_tensor,
    cat,
    reduce=True,
    alpha=1.0,
    return_cloud=False,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)

    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    chamferDist = ChamferDistance()
    if reduce:
        bidirectional_dist = chamferDist(
            target_pcd_tensor,
            src_pcd_tensor,
            bidirectional=True,
            alpha=alpha,
            robust=robust,
            delta=delta,
            bidirectional_robust=bidirectional_robust,
        )
    #
    else:
        bidirectional_dist = chamferDist(
            target_pcd_tensor,
            src_pcd_tensor,
            bidirectional=True,
            reduction=None,
            alpha=alpha,
            robust=robust,
            delta=delta,
            bidirectional_robust=bidirectional_robust,
        )
    #
    if return_cloud:
        return bidirectional_dist, target_pcd_tensor
    else:
        return bidirectional_dist


# compares pair loss of an input point cloud, with cloud generated from predicted params using EMD
def get_emd_loss_tensor(
    preds_tensor,
    src_pcd_tensor,
    cat,
    iterations=500
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    emd, _ = calc_emd(target_pcd_tensor, src_pcd_tensor, iterations=iterations)
    return emd*0.01



# compares pair loss of an input point cloud, with cloud generated from predicted params using EMD
def get_emd_loss_tensor(
    preds_tensor,
    src_pcd_tensor,
    cat,
    iterations=500
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)
    emd, _ = calc_emd(target_pcd_tensor, src_pcd_tensor, iterations=iterations)
    return emd*0.01


# this method compares pair loss of an input point cloud, with cloud generated from predicted params
def get_pair_loss_tensor(
    preds_tensor,
    src_pcd_tensor,
    cat
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)

    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    chamferDist = ChamferDistance()
    nn = chamferDist(
        src_pcd_tensor,
        target_pcd_tensor,
        bidirectional=True,
        return_nn=True
    )
    bidirectional_dist = torch.sum(nn[0].dists) + torch.sum(nn[1].dists)
    batch_size, point_count, _ = src_pcd_tensor.shape
    #print("s", bidirectional_dist)
    bidirectional_dist = bidirectional_dist / (batch_size * point_count)  
    true_idx_bwd = torch.gather(nn[1].idx, 1, nn[0].idx) # tgt[[src[match]]]
    
    paired_points_bwd = torch.stack([target_pcd_tensor[i][torch.flatten(nn[0].idx[i])] for i in range(nn[0].idx.shape[0])])
    pair_dist_bwd = paired_points_bwd - src_pcd_tensor
    paired_points_fwd = torch.stack([src_pcd_tensor[i][torch.flatten(true_idx_bwd[i])] for i in range(true_idx_bwd.shape[0])])
    pair_dist_fwd = paired_points_fwd - paired_points_bwd
    
    #print("sp", pair_dist_bwd.shape, pair_dist_fwd.shape)
    pair_dist = pair_dist_bwd + pair_dist_fwd
    pair_dist = torch.mul(torch.abs(pair_dist), torch.abs(pair_dist_bwd))
    pair_dist = torch.sum(pair_dist) / (batch_size * point_count)
    
    # paired_points_bwd = torch.stack([src_pcd_tensor[i][torch.flatten(true_idx_bwd[i])] for i in range(true_idx_bwd.shape[0])])
    # # paired_points_bwd = paired_points_bwd.reshape((paired_points_bwd.shape[0], 
    # #                                 paired_points_bwd.shape[1], 
    # #                                 paired_points_bwd.shape[3])) 
    # pair_dist_bwd = torch.sum(torch.square(paired_points_bwd - src_pcd_tensor))
    
    # true_idx_fwd = torch.gather(nn[0].idx, 1, nn[1].idx) # tgt[[src[match]]]
    # paired_points_fwd = torch.stack([target_pcd_tensor[i][torch.flatten(true_idx_fwd[i])] for i in range(true_idx_fwd.shape[0])])
    # # paired_points_fwd = paired_points_fwd.reshape((paired_points_fwd.shape[0], 
    # #                                 paired_points_fwd.shape[1], 
    # #                                 paired_points_fwd.shape[3])) 
    # pair_dist_fwd = torch.sum(torch.square(paired_points_fwd - target_pcd_tensor))
    
    # # pair_dist = (pair_dist_fwd + pair_dist_bwd) / (batch_size * point_count)
    # pair_dist = pair_dist_bwd / (batch_size * point_count)
    print("nn", bidirectional_dist.item(), pair_dist.item())
    return pair_dist + bidirectional_dist


# compute direction vectors of k neighbours
def knn_vectors(src_pcd_tensor, target_pcd_tensor, k):
    cuda = torch.device("cuda")

    # get nn
    chamferDist = ChamferDistance()
    nn = chamferDist(
        src_pcd_tensor, target_pcd_tensor, bidirectional=False, return_nn=True,
    k=k)
    nn = nn[0] # islolate forward direction

    # calculate directions
    N = src_pcd_tensor.shape[0] # batch size
    P1 = src_pcd_tensor.shape[1] # n_points in src cloud
    vectors = torch.zeros((N, P1, k, 3), device=cuda)

    # iterate through neighbours
    for i in range(k):
        diff = src_pcd_tensor - nn.knn[:,:,i,:]

        # normalise
        denom = torch.sqrt(torch.sum(torch.square(diff), 2))
        denom = denom.unsqueeze(2)
        vectors[:,:,i,:] = torch.div(diff, denom)

    #print(vectors.shape, nn.dists.shape)
    return vectors, nn.dists


# check if vectors are coplanar
# hardcoded to k=3 for now. TODO: support arbitrary dimensions
def check_coplanarity(vectors):
    cross = torch.cross(vectors[:,:,1,:], vectors[:,:,2,:], 2)
    #print(vectors[:,:,0,:].shape, cross.shape)
    dot = torch.einsum('ijk,ijk->ij', vectors[:,:,0,:], cross)

    #print(dot.shape, dot[0][:5], torch.max(dot))
    dot = torch.nan_to_num(dot)
    return(torch.absolute(dot))


def directional_chamfer_one_direction(src_pcd_tensor, target_pcd_tensor, k, direction_weight):
    vect, dists = knn_vectors(src_pcd_tensor, target_pcd_tensor, k)
    coplanarity = check_coplanarity(vect)
    dists = dists[:, :, 0] # isolate nearest neighbour

    #print("shapes", coplanarity.shape, dists.shape, (src_pcd_tensor.shape))
    dists = dists * (1-direction_weight) + dists * coplanarity * direction_weight
    dists = torch.sum(torch.sum(dists, dim=1), dim=0)
    dists = dists/coplanarity.shape[0]
    return dists


# chamfer loss, weighted by the coplanarity of knn points
# i.e. if multiple neighbours are coplanar with query point, they are weighted less
# this method compares an input point cloud, with cloud generated from predicted params
def get_chamfer_loss_directional_tensor(
    preds_tensor,
    src_pcd_tensor,
    cat,
    alpha=1.0,
    return_cloud=False,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
    k=1,
    direction_weight=0.2,
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)

    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    # compute loss
    forward_dist = directional_chamfer_one_direction(src_pcd_tensor, target_pcd_tensor, k, direction_weight)
    backward_dist = directional_chamfer_one_direction(target_pcd_tensor, src_pcd_tensor, k, direction_weight)
    bidirectional_dist  = alpha*forward_dist + backward_dist
    # chamferDist = ChamferDistance()
    # bidirectional_dist = chamferDist(
    #     target_pcd_tensor,
    #     src_pcd_tensor,
    #     bidirectional=False,
    #     reduction=None,
    #     alpha=alpha,
    #     robust=robust,
    #     delta=delta,
    #     bidirectional_robust=bidirectional_robust,
    # )

    #print("bidirectional_dist", bidirectional_dist.shape)
    #
    if return_cloud:
        return bidirectional_dist, target_pcd_tensor
    else:
        return bidirectional_dist


# compute mahalanobis distance between a set of point clouds and a mixture of gaussians
# specifically, distance is computed against each gaussian in the mixture, and the minimum distance is used
def mahalanobis_distance_gmm(
    target_pcd_tensor, means, covariances, robust=None, delta=0.1, weights=None
):
    # print("inputs", target_pcd_tensor.shape, means.shape, covariances.shape)
    # (b, n, 3), (b, 100, 3)
    dists = torch.zeros(
        (target_pcd_tensor.shape[0], target_pcd_tensor.shape[1], means.shape[1])
    ).cuda()

    # iterate through gaussians in the mixture
    for g in range(means.shape[1]):
        # compute mahalanobis distance between the points in the target cloud and the gaussian
        means_reshaped = (
            means[:, g, :].unsqueeze(1).repeat(1, target_pcd_tensor.shape[1], 1)
        )
        covariances_inv = torch.inverse(covariances)
        # print("inv", covariances_inv.shape)
        covariances_reshaped = (
            covariances_inv[:, g, :, :]
            .unsqueeze(1)
            .repeat(1, target_pcd_tensor.shape[1], 1, 1)
        )
        # print("reshaped", means_reshaped.shape, covariances_reshaped.shape)

        diff = target_pcd_tensor - means_reshaped
        # print("diff", diff.shape, torch.matmul(covariances_reshaped, dedifflta.unsqueeze(3)).shape, diff.view(diff.shape[0], diff.shape[1], 1, diff.shape[2]).shape)
        d = torch.matmul(
            diff.view(diff.shape[0], diff.shape[1], 1, diff.shape[2]),
            torch.matmul(covariances_reshaped, diff.unsqueeze(3)),
        )
        d = d.squeeze()
        # print("d", d.shape)
        dists[:, :, g] = d

    # find minimum distance for each point in the clouds
    min_dists, _ = torch.min(dists, dim=2)
    min_dists = torch.square(min_dists)/1000000
    #print("min dists", dists.shape, min_dists.shape, torch.sum(min_dists, dim=1).shape)

    # reduce
    return torch.sum(min_dists, dim=1)


def get_mahalanobis_loss_tensor(
    preds_tensor,
    means,
    covariances,
    cat,
    return_cloud=False,
    robust=None,
    delta=0.1,
    chamfer=0,
    alpha=1,
    src_pcd_tensor=None,
    weights=None,
):
    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    dist = mahalanobis_distance_gmm(
        target_pcd_tensor, means, covariances, robust=robust, delta=delta
    )
    dist = torch.sum(dist)

    if chamfer > 0:
        src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
        chamferDist = ChamferDistance()
        bidirectional_dist = chamferDist(
            src_pcd_tensor,
            target_pcd_tensor,
            bidirectional=False,
            robust=robust,
            delta=delta,
            alpha=alpha,
            #weights = weights,
        )
        print(dist, chamfer * bidirectional_dist)
        dist += chamfer * bidirectional_dist

    if return_cloud:
        return dist, target_pcd_tensor
    else:
        return dist


# this method compares the cloud generated from input params with the cloud generated from predicted params
def get_chamfer_loss_from_param_tensor(preds_tensor, src_tensor, cat):
    if cat == "elbow":
        target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
        src_pcd_tensor = generate_elbow_cloud_tensor(src_tensor)
    elif cat == "pipe":
        target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
        src_pcd_tensor = generate_pipe_cloud_tensor(src_tensor)
    elif cat == "tee":
        target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor, bp=True)
        src_pcd_tensor = generate_tee_cloud_tensor(src_tensor, bp=True)
    elif cat == "flange":
        target_pcd_tensor = generate_flange_cloud_tensor(preds_tensor, disc=True)
        src_pcd_tensor = generate_flange_cloud_tensor(
            src_tensor, disc=True
        )  # t2 = time.perf_counter()
    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(
        target_pcd_tensor, src_pcd_tensor, bidirectional=True
    )
    # t3 = time.perf_counter()
    # print("cloud", t2-t1, "chamf", t3-t2)
    return bidirectional_dist


# this method compares the cloud generated from input params with the cloud generated from predicted params.
# instead of searching for the nearest neighbour, it assumes an ordered correspondence between the points in the two clouds.
def get_correspondence_loss_from_param_tensor(preds_tensor, src_tensor, cat):
    if cat == "elbow" or "socket":
        target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
        src_pcd_tensor = generate_elbow_cloud_tensor(src_tensor)
    elif cat == "pipe":
        target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
        src_pcd_tensor = generate_pipe_cloud_tensor(src_tensor)
    elif cat == "tee":
        target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor, bp=True)
        src_pcd_tensor = generate_tee_cloud_tensor(src_tensor, bp=True)
    elif cat == "flange":
        target_pcd_tensor = generate_flange_cloud_tensor(preds_tensor, disc=True)
        src_pcd_tensor = generate_flange_cloud_tensor(
            src_tensor, disc=True
        )
    l2_loss = torch.sum(torch.square(target_pcd_tensor - src_pcd_tensor), dim=(1, 2))
    l2_loss = l2_loss.mean()

    # chamferDist = ChamferDistance()
    # bidirectional_dist = chamferDist(
    #     target_pcd_tensor, src_pcd_tensor, bidirectional=True
    # )
    #print("l2", l2_loss, "chamf", bidirectional_dist)
    return l2_loss


# this method compares an input point cloud, with a second point cloud
def get_cloud_chamfer_loss_tensor(
    src_pcd_tensor,
    tgt_pcd_tensor,
    alpha=1.0,
    separate_directions=False,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
    reduction=None
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    tgt_pcd_tensor = tgt_pcd_tensor.transpose(2, 1)

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(
        tgt_pcd_tensor,
        src_pcd_tensor,
        bidirectional=True,
        reduction=reduction,
        separate_directions=separate_directions,
    )
    if separate_directions == True:
        bidirectional_dist = torch.cat(
            [
                torch.unsqueeze(bidirectional_dist[0], dim=-1),
                torch.unsqueeze(bidirectional_dist[1], dim=-1),
            ],
            1,
        )

    return bidirectional_dist


def farthest_point_sample_gpu(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    cuda = torch.device("cuda")
    N, D = point.shape
    xyz = point[:, :3]
    centroids = torch.zeros((npoint,), device=cuda, dtype=torch.long)
    distance = torch.ones((N,), device=cuda, dtype=torch.double) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids



def get_pair_loss_clouds_tensor(x, y, k=1, add_pair_loss=True, it=0, return_assignment=True):
    cuda = torch.device("cuda")
    
    chamferDist = ChamferDistance()
    if not add_pair_loss:
        if k==1:
            bidirectional_dist = chamferDist(
                x,
                y,
                bidirectional=True,
                reduction="mean",
                separate_directions=False,
                robust=None
            )
        else:
            nn = chamferDist(x, y, bidirectional=True, return_nn=True, k=k)
            batch_size, point_count, _ = x.shape
            bidirectional_dist = torch.sum(nn[0].dists) + torch.sum(nn[1].dists)
            bidirectional_dist = bidirectional_dist / (batch_size * point_count)
    else:
        # add a loss term for mismatched pairs
        k = 3
        nn = chamferDist(
            x, y, bidirectional=True, return_nn=True, k=k
        )
        #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
        #bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:,:,0])
        bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) 
        batch_size, point_count, _ = x.shape
        # print("s", nn[0].idx.shape, nn[1].idx.shape)
        
        idx_fwd = torch.unsqueeze(nn[0].idx[:, :, 0], 2).repeat(1, 1, k)
        idx_bwd = torch.unsqueeze(nn[1].idx[:, :, 0], 2).repeat(1, 1, k)
        # print("f", idx_fwd.shape)
        true_idx_fwd = torch.gather(idx_fwd, 1,  nn[1].idx) # tgt[[src[match]]]
        true_idx_bwd = torch.gather(idx_bwd, 1, nn[0].idx) # tgt[[src[match]]]
        # print("t", true_idx_fwd[0,:2], nn[0].idx[0,649], nn[1].idx[0,:2])
        
        # manual chamfer loss
        paired_points_x_to_y = torch.stack([y[i][nn[0].idx[i]] for i in range(nn[0].idx.shape[0])])
        # print("p", paired_points_x_to_y.shape, torch.unsqueeze(x, 2).repeat(1, 1, k, 1).shape)
        pair_dist_x_to_y = paired_points_x_to_y - torch.unsqueeze(x, 2).repeat(1, 1, k, 1)

        paired_points_y_to_x = torch.stack([x[i][true_idx_bwd[i]] for i in range(true_idx_bwd.shape[0])])
        # print("p2", paired_points_y_to_x.shape, paired_points_x_to_y.shape)
        pair_dist_y_to_x = paired_points_y_to_x - paired_points_x_to_y

        pair_dist = torch.sum(torch.square(pair_dist_x_to_y + pair_dist_y_to_x), 3)
        mdb, min_idx_bwd = torch.min(pair_dist, 2)
        #print("p3", mdb.shape, mdb[0,:5], min(mdb[0]), torch.count_nonzero(mdb[0]))
        
        # select the best neighbour of x in y (nn[0]) such that the x->y->x distance is minimized
        min_dist_bwd = torch.gather(nn[0].dists, 2, min_idx_bwd.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        #print("p4", min_dist_bwd.shape) 
        
        
        # reverse
        rpaired_points_y_to_x = torch.stack([x[i][nn[1].idx[i]] for i in range(nn[1].idx.shape[0])])
        # print("p", paired_points_x_to_y.shape, torch.unsqueeze(x, 2).repeat(1, 1, k, 1).shape)
        rpair_dist_y_to_x = rpaired_points_y_to_x - torch.unsqueeze(y, 2).repeat(1, 1, k, 1)

        rpaired_points_x_to_y = torch.stack([y[i][true_idx_fwd[i]] for i in range(true_idx_fwd.shape[0])])
        # print("p2", paired_points_y_to_x.shape, paired_points_x_to_y.shape)
        rpair_dist_x_to_y = rpaired_points_x_to_y - rpaired_points_y_to_x

        rpair_dist = torch.sum(torch.square(rpair_dist_y_to_x + rpair_dist_x_to_y), 3)
        mdf, min_idx_fwd = torch.min(rpair_dist, 2)
        #print("p5", min_idx_fwd.shape, nn[1].dists.shape)
        
        # select the best neighbour of x in y (nn[0]) such that the x->y->x distance is minimized
        min_dist_fwd = torch.gather(nn[1].dists, 2, min_idx_fwd.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        #print("p6", min_dist_fwd.shape)
        
        pair_distance = torch.sum(min_dist_bwd) #+ torch.sum(min_dist_fwd)
        #pair_dist = torch.sum(mdf) + torch.sum(mdb)
       
        #print(pair_dist)
        # pair_dist = torch.mul(torch.abs(pair_dist), torch.abs(pair_dist_x_to_y))
        # pair_dist = torch.sum(pair_dist)
        #pair_dist = torch.sum(torch.square(pair_dist))
        
        # reverse direction
        # reverse_paired_points_y_to_x = torch.stack([x[i][torch.flatten(nn[1].idx[i])] for i in range(nn[1].idx.shape[0])])
        # reverse_pair_dist_y_to_x = reverse_paired_points_y_to_x - y

        # reverse_paired_points_x_to_y = torch.stack([y[i][torch.flatten(true_idx_fwd[i])] for i in range(true_idx_fwd.shape[0])])
        # reverse_pair_dist_x_to_y = reverse_paired_points_x_to_y - reverse_paired_points_y_to_x

        # reverse_pair_dist = reverse_pair_dist_y_to_x + reverse_pair_dist_x_to_y
        # reverse_pair_dist = torch.mul(torch.abs(reverse_pair_dist), torch.abs(reverse_pair_dist_y_to_x))
        # reverse_pair_dist = torch.sum(reverse_pair_dist)
        #reverse_pair_dist = torch.sum(torch.square(reverse_pair_dist))
        
        # pair_dist += reverse_pair_dist
        print("manual", (torch.sum(torch.square(rpair_dist_y_to_x[:,:,0,:])) + torch.sum(torch.square(pair_dist_x_to_y[:,:,0,:]))).item())
        print("dist", bidirectional_dist.item(), pair_distance.item())
        bidirectional_dist = bidirectional_dist + pair_distance
        #bidirectional_dist = pair_distance 
        bidirectional_dist = bidirectional_dist / (batch_size)
        
            
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, min_idx_fwd.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, min_idx_bwd.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        return bidirectional_dist, [min_ind_0, min_ind_1]
    
    else:
        return bidirectional_dist


# add jitter to the point correspondences in CD
# NOTE: hard coded for batch_size = 1 only
def get_jittery_cd_tensor(x, y, k=1, it=0):
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()

    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=1
    )
    #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
    bidirectional_dist = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
    batch_size, point_count, _ = x.shape
    
    fps= False

    #jitter_size = int(64*(0.001*(1000-it)))+1
    jitter_size = 2
    print("jitter", jitter_size)
    perm1 = torch.randperm(x.size(1), device=cuda)[:jitter_size]
    perm2 = torch.randperm(x.size(1), device=cuda)[:jitter_size].unsqueeze(1)
    nn_copy = nn[0].idx.clone()
    
    if fps:
        # farthest point sample
        centroids  = farthest_point_sample_gpu(x[0], jitter_size)
        nn_copy[0][perm1] = centroids.unsqueeze(1)
    else:   
        # randomly permute
        for cloud in nn_copy:
            cloud[perm1] = perm2
    paired_points_x_to_y = torch.stack([y[i][torch.flatten(nn_copy[i])] for i in range(nn_copy.shape[0])])
    pair_dist_x_to_y = paired_points_x_to_y - x
    
    # reverse
    rperm1 = torch.randperm(y.size(1), device=cuda)[:jitter_size]
    rperm2 = torch.randperm(y.size(1), device=cuda)[:jitter_size].unsqueeze(1)
    rnn_copy = nn[1].idx.clone()
            
    if fps:
        # farthest point sample
        rcentroids  = farthest_point_sample_gpu(y[0], jitter_size)
        rnn_copy[0][rperm1] = rcentroids.unsqueeze(1)
    else:
        #randomly permute
        for cloud in rnn_copy:
            cloud[rperm1] = rperm2
    rpaired_points_x_to_y = torch.stack([x[i][torch.flatten(rnn_copy[i])] for i in range(rnn_copy.shape[0])])
    rpair_dist_x_to_y = rpaired_points_x_to_y - y
    
    
    pair_dist = torch.sum(torch.square(pair_dist_x_to_y)) + torch.sum(torch.square(rpair_dist_x_to_y))
    
    print("dist", bidirectional_dist.item(), pair_dist.item())
    #bidirectional_dist = bidirectional_dist + pair_dist
    bidirectional_dist = pair_dist 
    bidirectional_dist = bidirectional_dist / (batch_size)
        
    return bidirectional_dist


# add self loss to CD
def get_self_cd_tensor(x, y, thresh=0.001):
    cuda = torch.device("cuda")
    
    chamferDist = ChamferDistance()

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=1
    )
    #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
    bidirectional_dist = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
    batch_size, point_count, _ = x.shape
    
    # compute self loss for gen cloud
    nn2 = chamferDist(y, y, bidirectional=False, return_nn=True, k=2)
    self_loss = torch.sum(torch.square(torch.clamp(thresh - nn2[0].dists[:,:,1], min=0)))
    #self_loss = torch.sum(torch.square((torch.abs(nn[0].dists[:,:,0] - nn2[0].dists[:,:,1]))))

    print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = bidirectional_dist + self_loss*1000
    #bidirectional_dist = pair_dist 
    bidirectional_dist = bidirectional_dist / (batch_size)
        
    return bidirectional_dist


# compute reverse weighted chamfer loss
# this is computed by calclating nn at a large k, then scaling each correspondences's 
# distance by the reverse CD of that correspondence. The minimum of these is used to index 
# the coorespondence to be chosen for measuring chamfer distance.
# In other words, whenever a point in cloud B already has a close correspondence in cloud A,
# it becomes less attractive to other points in cloud A, pushing points in cloud A to find 
# other correspondences.
def calc_reverse_weighted_cd_tensor(x, y, k=32, return_assignment=False):   
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    pow = 2
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # get closest distances in reverse direction
    scaling_factors_1 = nn[0].dists[:,:,0].unsqueeze(2).repeat(1, 1, k)
    denominator_1 = torch.pow(torch.gather(scaling_factors_1, 1, nn[1].idx), pow)
    #denominator_1 = torch.gather(scaling_factors_1, 1, nn[1].idx)
    # divide by closest distance in reverse direction, selectfind minimum
    scaled_dist_1 = torch.div(nn[1].dists, denominator_1)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
    #scaled_dist_1x = scaled_dist_1x - torch.ones_like(scaled_dist_1x)
    #print("d", torch.min(scaled_dist_1x[0]))
    
    
    #min_dist_1 = torch.stack([nn[1].dists[0][i][i1[0][i]] for i in range(nn[1].dists[0].shape[0])]).unsqueeze(0)
    # select distance that corresponds to above minimum index
    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    #print("s", min_dist_1[0][100], nn[1].dists[0][100])

    # reverse direction
    scaling_factors_0 = nn[1].dists[:,:,0].unsqueeze(2).repeat(1, 1, k)
    #denominator_0 = torch.gather(scaling_factors_0, 1, nn[0].idx)
    denominator_0 = torch.pow(torch.gather(scaling_factors_0, 1, nn[1].idx), pow)
    scaled_dist_0 = torch.div(nn[0].dists, denominator_0)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    #scaled_dist_0x = scaled_dist_0x - torch.ones_like(scaled_dist_0x)
    #print(i2.shape)
    #min_dist_0 = torch.stack([nn[0].dists[0][i][i2[0][i]] for i in range(nn[0].dists[0].shape[0])]).unsqueeze(0)
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
 
    #bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    #self_loss = torch.sum(scaled_dist_1x) + torch.sum(scaled_dist_0x)
    self_loss = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    #bidirectional_dist = bidirectional_dist #+ self_loss
    bidirectional_dist = self_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


# weight the distance of each correspondence by the distances to all its correspondences
def calc_neighbour_weighted_cd_tensor(x, y, k=32, return_assignment=True):   
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # compile a list of points in y that correspond to x
    # NOTE: only for batch size = 1
    # sum 1/ all distances from y that correspond to x for each point in x
    # nn[0] is from x to y, nn[1] is from y to x 
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    mask = (nn[1].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0))

    # Calculate values for each index using the mask
    values = torch.where(mask, 1/nn[1].dists[0, :, 0], torch.tensor(0., device=cuda))

    # Sum along the appropriate dimension to get the final dists_x
    dists_x = torch.sum(values, dim=0)
    # print(dists_x.shape, dists_x[:5])

    dists_x = (dists_x + 1.).unsqueeze(0)

    # scale all distances by the scaling factor
    scaling_factors_1 = dists_x.unsqueeze(2).repeat(1, 1, k)
    denominator_1 = torch.gather(scaling_factors_1, 1, nn[1].idx)
    # divide by closest distance in reverse direction, select minimum
    scaled_dist_1 = torch.mul(nn[1].dists, denominator_1)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)

    #print("den", dists_x[0][:5])
    # select distance that corresponds to above minimum index
    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    #print("s", min_dist_1[0][100], nn[1].dists[0][100])


    # reverse direction
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    mask = (nn[0].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0))

    # Calculate values for each index using the mask
    values = torch.where(mask, 1/nn[0].dists[0, :, 0], torch.tensor(0., device=cuda))

    # Sum along the appropriate dimension to get the final dists_x
    dists_y = torch.sum(values, dim=0)
    dists_y = (dists_y + 1.).unsqueeze(0)
    
    scaling_factors_0 = dists_y.unsqueeze(2).repeat(1, 1, k)
    denominator_0 = torch.gather(scaling_factors_0, 1, nn[0].idx)
    scaled_dist_0 = torch.mul(nn[0].dists, denominator_0)
    #])
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    #scaled_dist_0x = scaled_dist_0x - torch.ones_like(scaled_dist_0x)
    #print(i2.shape)
    #min_dist_0 = torch.stack([nn[0].dists[0][i][i2[0][i]] for i in range(nn[0].dists[0].shape[0])]).unsqueeze(0)
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
 
    # print("d", nn[0].dists[:,:5,0], scaled_dist_0x[0, :5])
    # print("d2", nn[1].dists[:,:5,0], scaled_dist_1x[0, :5], scaled_dist_1[0, :5], denominator_1[0, :5], nn[1].dists[0:,:5])
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    
    self_loss = torch.sum(scaled_dist_1x) + torch.sum(scaled_dist_0x)
    #self_loss = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    #bidirectional_dist = bidirectional_dist #+ self_loss
    bidirectional_dist = self_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


def get_reverse_weighted_cd_tensor(preds_tensor, src_pcd_tensor, cat, k=32):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    dist = calc_reverse_weighted_cd_tensor(target_pcd_tensor, src_pcd_tensor, k=k, 
                                           return_assignment=False)
    return dist


# for each point Ai, measure the probability of being pared with each point in the other cloud B
# as a continuous function going to zero at k.
# P is thresholded by max and min values of the value matrix
# do the same for cloud B
# loss = SUM( P(Ai->Bj) * (1 - P(Bj->Ai))) and vice versa 
def calc_pairing_probabilty_loss_tensor(x, y, k=32, return_assignment=True):
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # compile a list of points in y that correspond to x
    # NOTE: only for batch size = 1
    # sum 1/ all distances from y that correspond to x for each point in x
    # nn[0] is from x to y, nn[1] is from y to x 
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    # print("a", nn[1].idx[0, :, 0].unsqueeze(1).shape, torch.arange(nn[0].idx.shape[1]).unsqueeze(0).shape)
    # print("a2", nn[1].idx[0].shape, torch.arange(nn[0].idx.shape[1]).unsqueeze(0).repeat(k,1).shape)
    # print("a3", nn[0].idx[0, :, 0].unsqueeze(1).shape, torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0).shape)
    # print("b", nn[1].dists[0,:,0].shape)
    
    
    # probs_x = torch.zeros((nn[0].idx.shape[1], nn[1].idx.shape[1]), device=cuda)
    # for i in range(nn[0].idx.shape[1]):
    #     values = torch.where(nn[1].idx[0] == i, 1/nn[1].dists[0], 0.01)
    #     #print("v", values.shape, torch.sum(values, 1).shape)
    #     #print(dists_x.shape, torch.sum(values, 1).shape)
    #     probs_x[i] = torch.sum(values, 1)
    
    idx = torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(1).unsqueeze(2)
    probs_x = torch.where(nn[1].idx[0] == idx, 1/nn[1].dists[0], 0.01)
    probs_x = torch.sum(probs_x, 2)
    #print("p", probs_x.shape)
    #probs_x = probs_x / torch.max(probs_x)
    probs_x = F.normalize(probs_x, p=2, dim=0)
    #print(probs_x.shape, torch.count_nonzero(probs_x, 1))
 
    idx = torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(1).unsqueeze(2)
    probs_y = torch.where(nn[0].idx[0] == idx, 1/nn[0].dists[0], 0.01)
    probs_y = torch.sum(probs_y, 2)
    #probs_y = probs_y / torch.max(probs_y)
    probs_y = F.normalize(probs_y, p=2, dim=0)
    #print(torch.max(probs_y, dim=0))


    # mask = (nn[1].idx[0] == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0).repeat(k,1))
    # #mask = (nn[1].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0))
    # print("m", mask.shape, torch.count_nonzero(mask))

    # # Calculate values for each index using the mask
    # probs_x = torch.where(mask, 1/nn[1].dists[0], torch.tensor(0., device=cuda))
    # print("nz", probs_x[:2], torch.count_nonzero(probs_x, 0))
    # # normalise. farthest distance (out of knn range) = 0, closest = 1
    # probs_x = probs_x / torch.max(probs_x)
    # # print("values", probs_x.shape, torch.min(probs_x), torch.max(probs_x))

    # # reverse direction
    # # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    # mask = (nn[0].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0))

    # # Calculate values for each index using the mask
    # probs_y = torch.where(mask, 1/nn[0].dists[0, :, 0], torch.tensor(0., device=cuda))
    # # normalise. farthest distance (out of knn range) = 0, closest = 1
    # probs_y = probs_y / torch.max(probs_y)
    
    
    #print("values", probs_y.shape, torch.min(probs_y), torch.max(probs_y))
    #probability_loss = torch.sum(torch.mul(probs_x, (1 - torch.transpose(probs_y, 0, 1)))) + torch.sum(torch.mul(probs_y, (1 - torch.transpose(probs_x, 0, 1))))
    probability_loss = torch.sum(1. - torch.mul(probs_x, torch.transpose(probs_y, 0, 1))) #+ torch.sum(1 - torch.mul(probs_y, torch.transpose(probs_x, 0, 1)))
    #probability_loss = -1*torch.sum(torch.mul(probs_x, torch.transpose(probs_y, 0, 1))) #+ torch.sum(1 - torch.mul(probs_y, torch.transpose(probs_x, 0, 1)))
    probability_loss = probability_loss*0.1
    
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])

    batch_size, point_count, _ = x.shape

    # print("dist", bidirectional_dist.item(), probability_loss.item())
    bidirectional_dist = bidirectional_dist + probability_loss
    #bidirectional_dist = probability_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        return bidirectional_dist, [nn[0].idx[0,:,0], nn[1].idx[0,:,0]]
    return bidirectional_dist


# aside from matching by shortest distance, also match by density around each point
# density for each point is measured by the sum of its distances to its k neighbours in the same cloud
def calc_balanced_chamfer_loss_tensor(x, y, k=32, return_assignment=False):
    chamferDist = ChamferDistance()
    eps = 0.00001

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    
    k2 = 32 # reduce k to check density in smaller patches
    power = 8
    
    # measure density with itself
    nn_x = chamferDist(x, x, bidirectional=False, return_nn=True, k=k2)
    density_x = torch.mean(nn_x[0].dists[:,:,1:], dim=2)
    density_x = 1 / (density_x + eps)
    high, low = torch.max(density_x), torch.min(density_x)
    diff = high - low
    density_x = (density_x - low) / diff
    
    # measure density with other cloud
    density_xy = torch.mean(nn[0].dists[:,:,:k2-1], dim=2)
    density_xy = 1 / (density_xy + eps)
    high, low = torch.max(density_xy), torch.min(density_xy)
    diff = high - low
    density_xy = (density_xy - low) / diff
    w_x = torch.div(density_xy, density_x)
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_1 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_1, 1, nn[1].idx)
    
    scaled_dist_1 = torch.mul(nn[1].dists, multiplier)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
        
    # measure density with itself
    nn_y = chamferDist(y, y, bidirectional=False, return_nn=True, k=k2)
    density_y = torch.mean(nn_y[0].dists[:,:,1:], dim=2)
    density_y = 1 / (density_y + eps)
    high, low = torch.max(density_y), torch.min(density_y)
    diff = high - low
    density_y = (density_y - low) / diff
    
    # measure density with other cloud
    density_yx = torch.mean(nn[1].dists[:,:,:k2-1], dim=2)
    density_yx = 1 / (density_yx + eps)
    high, low = torch.max(density_yx), torch.min(density_yx)
    diff = high - low
    density_yx = (density_yx - low) / diff
    w_x = torch.div(density_yx, density_y)
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_0 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_0, 1, nn[0].idx)
    
    scaled_dist_0 = torch.mul(nn[0].dists, multiplier)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    
    #print("d", w_x.shape, i1.shape)
    # reverse

    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    
    balanced_cd = torch.sum(torch.sqrt(min_dist_1)) + torch.sum(torch.sqrt(min_dist_0))
    #balanced_cd = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    #balanced_cd = torch.sum(min_dist_1) + torch.sum(nn[0].dists[:, :, 0])
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    # print("cd", torch.sum(nn[1].dists[:,:,0]).item(), torch.sum(nn[0].dists[:, :, 0]).item())
    # print("balanced", torch.sum(min_dist_1).item(), torch.sum(min_dist_0).item())
    bidirectional_dist = balanced_cd
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        min_ind_0 = nn[0].idx[:,:,0]

        return bidirectional_dist, [min_ind_0.detach().cpu().numpy(), min_ind_1.detach().cpu().numpy()]
    else:
        return bidirectional_dist


def get_balanced_chamfer_loss_tensor(preds_tensor, src_pcd_tensor, cat, k=32):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    target_pcd_tensor = get_shape_cloud_tensor(preds_tensor, cat)

    dist = calc_balanced_chamfer_loss_tensor(target_pcd_tensor, src_pcd_tensor, k=k, 
                                           return_assignment=False)
    return dist


# aside from matching by shortest distance, also match by density around each point
# density for each point is measured by the sum of its distances to its k neighbours in the same cloud
def calc_balanced_single_chamfer_loss_tensor(x, y, k=32, return_assignment=False):
    chamferDist = ChamferDistance()
    eps = 0.00001

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    
    k2 = 32 # reduce k to check density in smaller patches
    power = 8
    # measure density with itself
    # nn_x = chamferDist(x, x, bidirectional=False, return_nn=True, k=k2)
    # density_x = torch.mean(nn_x[0].dists[:,:,1:], dim=2)
    # density_x = 1 / (density_x + eps)
    # high, low = torch.max(density_x), torch.min(density_x)
    # diff = high - low
    # density_x = (density_x - low) / diff
    
    # measure density with other cloud
    density_xy = torch.mean(nn[0].dists[:,:,:k2], dim=2)
    density_xy = 1 / (density_xy + eps)
    high, low = torch.max(density_xy), torch.min(density_xy)
    diff = high - low
    density_xy = (density_xy - low) / diff
    #w_x = torch.div(density_xy, density_x)
    w_x = density_xy
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_1 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_1, 1, nn[1].idx)
    
    scaled_dist_1 = torch.mul(nn[1].dists, multiplier)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
        
    # measure density with itself
    # nn_y = chamferDist(y, y, bidirectional=False, return_nn=True, k=k2)
    # density_y = torch.mean(nn_y[0].dists[:,:,1:], dim=2)
    # density_y = 1 / (density_y + eps)
    # high, low = torch.max(density_y), torch.min(density_y)
    # diff = high - low
    # density_y = (density_y - low) / diff
    
    # measure density with other cloud
    density_yx = torch.mean(nn[1].dists[:,:,:k2], dim=2)
    density_yx = 1 / (density_yx + eps)
    high, low = torch.max(density_yx), torch.min(density_yx)
    diff = high - low
    density_yx = (density_yx - low) / diff
    #w_x = torch.div(density_yx, density_y)
    w_x = density_yx
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_0 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_0, 1, nn[0].idx)
    
    scaled_dist_0 = torch.mul(nn[0].dists, multiplier)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    
    #print("d", w_x.shape, i1.shape)
    # reverse

    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    
    balanced_cd = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    #balanced_cd = torch.sum(min_dist_1) + torch.sum(nn[0].dists[:, :, 0])
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    #print("dist, balanced", bidirectional_dist.item(), balanced_cd.item())
    bidirectional_dist = balanced_cd
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = nn[0].idx[0,:,0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


def calc_cd_like_InfoV2(x, y, return_assignment=False):
    chamferDist = ChamferDistance()
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True)
        
    dist1, dist2, idx1, idx2 = nn[0].dists, nn[1].dists, nn[0].idx, nn[1].idx
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
    distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    if return_assignment:
        return (torch.sum(distances1) + torch.sum(distances2)) / 2, [idx1.detach().cpu().numpy(), 
                                                                     idx2.detach().cpu().numpy()]
    return (torch.sum(distances1) + torch.sum(distances2)) / 2


# get any of chamfer, EMD, reverse or jittered chamfer loss
# NOTE: for EMD, gradient is only calculated for y, not x
def calculate_3d_loss(x, y, loss_funcs, it=0, batch_size=None):

    losses = {}
    for loss_func in loss_funcs:
            if loss_func == "chamfer":
                chamferDist = ChamferDistance()
                losses[loss_func] = chamferDist(x, y, bidirectional=True).item()
            elif loss_func == "emd":
                losses[loss_func] = calc_emd(y, x)[0].item()
            elif loss_func == "reverse":
                losses[loss_func] = calc_reverse_weighted_cd_tensor(x, y, return_assignment=False).item()
            elif loss_func == "jittery":
                losses[loss_func] = get_jittery_cd_tensor(x, y, it=it).item()
    
    return losses