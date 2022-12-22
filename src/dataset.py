import numpy as np
import math
import random
import os
import json
import torch
import copy

import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from path import Path

from src.preperation import *

def parse_tee_properties(element_data):
  #target = [element_data['radius']/1000, element_data['length']/1000]
  scaled_targets = [element_data['radius1']/1000, element_data['length1']/1000, 
                    element_data['radius2']/1000, element_data['length2']/1000]
  unscaled_targets = [element_data['direction1'][0], element_data['direction1'][1],
                      element_data['direction1'][2], element_data['direction2'][0], 
                      element_data['direction2'][1], element_data['direction2'][2],
                      element_data['position1'][0]/1000, element_data['position1'][1]/1000,
                      element_data['position1'][2]/1000]

  #target = [element_data['radius']/1000]
  return np.array(scaled_targets), np.array(unscaled_targets)


def parse_pipe_properties(element_data):
  scaled_targets = [element_data['radius']/1000, element_data['length']/1000]
  unscaled_targets = [element_data['direction'][0], element_data['direction'][1],
                      element_data['direction'][2]]

  return np.array(scaled_targets), np.array(unscaled_targets)


def parse_elbow_properties(element_data):
  #target = [element_data['radius']/1000, element_data['length']/1000]
  scaled_targets = [element_data['radius']/1000, element_data['axis_x']/1000, 
                    element_data['axis_y']/1000]
  unscaled_targets = [element_data['direction'][0], element_data['direction'][1], 
                      element_data['direction'][2],  element_data['angle']/200,
                      element_data['position'][0]/1000, element_data['position'][1]/1000,
                      element_data['position'][2]/1000]
  #target = [element_data['radius']/1000]
  return np.array(scaled_targets), np.array(unscaled_targets)


def default_transforms():
    return transforms.Compose([
                                Normalize(),
                                ToTensor()
                              ])


# scaled properties must be transformed when the cloud's scale is transformed
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", category='pipe', transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.category = category
        self.transforms = transform if not valid else default_transforms()
        metadata_file = open(root_dir/Path(category)/"metadata_new.json", 'r')
        metadata = json.load(metadata_file)
        self.valid = valid
        self.files = []

        new_dir = root_dir/Path(category)/folder
        for file in os.listdir(new_dir):
            if file.endswith('.pcd'):
                sample = {}
                sample['pcd_path'] = new_dir/file
                sample['id'] = int(file.split(".")[0])
                if category == 'pipe':
                    sample['scaled_properties'], sample['unscaled_properties'] = parse_pipe_properties(
                        metadata[file.split(".")[0]])
                elif category == 'elbow':
                    sample['scaled_properties'], sample['unscaled_properties'] = parse_elbow_properties(
                        metadata[file.split(".")[0]])
                elif category == 'tee':
                    sample['scaled_properties'], sample['unscaled_properties'] = parse_tee_properties(
                        metadata[file.split(".")[0]])
                self.files.append(sample)
        self.targets = len(self.files[0]['scaled_properties']) + len(
            self.files[0]['unscaled_properties'])

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, properties):
        cloud = read_pcd(file)
        if self.transforms:
            pointcloud, properties = self.transforms((cloud, properties))
        return pointcloud, properties

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        scaled_properties = self.files[idx]['scaled_properties']
        unscaled_properties = torch.from_numpy(self.files[idx]['unscaled_properties']).float()
        id = self.files[idx]['id']
        pointcloud, scaled_properties = self.__preproc__(pcd_path, scaled_properties)
        return {'pointcloud': pointcloud, 
                'properties': torch.cat((scaled_properties, unscaled_properties)),
                'id': id}