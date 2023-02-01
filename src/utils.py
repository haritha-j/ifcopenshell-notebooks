import numpy as np
import os
import torch
import pickle
import open3d as o3d

from src.chamfer import *
from src.visualisation import *


def scale_preds(preds, cat, up=1, norm_factor = 1):
    if up == 1:
        scale_factor = 1000
    elif up == 0:
        scale_factor = 0.001
    else:
        scale_factor = 1

    if cat == 'pipe':
        scalable_targets = [0,1]
    elif cat == 'elbow':  
        scalable_targets = [0,1,2]
    elif cat == 'tee':
        scalable_targets = [0,1,2,3]

    for j in scalable_targets:
        preds[j] = preds[j]*scale_factor*norm_factor
        
    #handle negative radius
    #preds[0] = abs(preds[0])
        
    return preds

def translate_preds(preds, cat, translation):
    if cat == 'tee':
        targets = [4,5,6]
    elif cat == 'elbow':
        targets = [3,4,5]
    elif cat == 'pipe':
        targets = [2,3,4]
        
    for i, t in enumerate(targets):
        preds[t] = preds[t] + translation[i]
    return preds


def prepare_visualisation(pcd_id, cat, cloud_id, cloud_list, predictions_list, path, ext):
    pcd_path = path + cat + "/test/" + str(pcd_id) + ext

    # load pcd and 'un-normalise'
    pcd_temp = o3d.io.read_point_cloud(pcd_path).points
    norm_pcd_temp = np.mean(pcd_temp, axis=0)
    pcd_temp -= norm_pcd_temp
    norm_factor = np.max(np.linalg.norm((pcd_temp), axis=1))
    points = (cloud_list[cloud_id].transpose(1, 0))*norm_factor #+ norm_pcd_temp
    pcd = o3d.utility.Vector3dVector(points)

    # scale predictions when necessary
    preds = predictions_list[cloud_id].tolist()
    preds = scale_preds(preds, cat, norm_factor=norm_factor)
    
    preds = [float(pred) for pred in preds]
    return pcd, preds


def undo_normalisation(pcd_id, cat, preds, path, ext):
    pcd_path = path + cat + "/test/" + str(pcd_id) + ext

    # load pcd and 'un-normalise'
    pcd_temp = o3d.io.read_point_cloud(pcd_path).points
    norm_pcd_temp = np.mean(pcd_temp, axis=0)
    pcd_temp -= norm_pcd_temp
    norm_factor = np.max(np.linalg.norm((pcd_temp), axis=1))

    # scale and translate predictions 
    preds = scale_preds(preds, cat, up=2, norm_factor=norm_factor)
    preds = translate_preds(preds, cat, norm_pcd_temp)
    preds = [float(pred) for pred in preds]
    return preds


def load_preds(preds_dir, cat):
    preds_file = preds_dir + 'preds_' + cat + '.pkl'
    with open(preds_file, 'rb') as f:
        return pickle.load(f)
    

 # visualise all predictions together   
def batch_visualise(preds_dir, blueprint, path, ext, device, ifc = True):
    # load predictions
    all_dists = []
    for cat in ['tee', 'elbow', 'pipe']:
    #for cat in ['tee']:
        preds, ids, dists = load_preds(preds_dir, cat)
        print(cat, len(preds))
        all_dists.append(dists)

        # generate ifc models for predictions
        if ifc:
            scaled_preds = [scale_preds(p.tolist(), cat) for p in preds]
            ifc = visualize_predictions(None, cat, scaled_preds, blueprint, visualize=False)
            ifc.write(preds_dir + cat + "_bp.ifc")
    
        # generate point clouds for predictions
        else:
            original_preds = []
            for i in range(len(preds)):
                pcd_id = ids[i].item()
                original_preds.append(undo_normalisation(pcd_id, cat, preds[i], path, ext))
            preds_tensor = torch.Tensor(original_preds).to(device)
            
            if cat == "elbow":
                target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
            elif cat == "pipe":
                target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
            elif cat == "tee":
                target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor)
           
            points = target_pcd_tensor.cpu().detach().numpy()
            points = np.reshape(points, (points.shape[0]*points.shape[1], points.shape[2]))
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(preds_dir + cat + "_bp.pcd", cloud)
            
            
def merge_clouds(directory, cat):
    path = directory + cat + "/test/"
    files = os.listdir(path)
    points_lists = []
    for f in files:
        if f.split(".")[-1] == "pcd" or f.split(".")[-1] == "ply":
            pcd_path = path + f
            points_lists.append(o3d.io.read_point_cloud(pcd_path).points)
    
    all_points = np.concatenate(points_lists)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud(cat+"_bp_input.pcd", cloud)