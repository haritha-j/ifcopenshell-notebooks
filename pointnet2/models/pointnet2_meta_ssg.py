import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as trnsfrm

from pointnet2_utils import PointNetSetAbstraction
from src.chamfer import get_cloud_chamfer_loss_tensor

class get_model(nn.Module):
    def __init__(self,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
    #    # TODO: delete the following variables before retraining
    #     self.fc1 = nn.Linear(1024, 512)
    #     self.bn1 = nn.BatchNorm1d(512)
    #     self.drop1 = nn.Dropout(0.4)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.bn2 = nn.BatchNorm1d(256)
    #     self.drop2 = nn.Dropout(0.4)
    #     self.fc3 = nn.Linear(256, 14)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, points, points_trans, registration=False, loss_type="chamfer"):
        if not registration:
            chamfer_loss = get_cloud_chamfer_loss_tensor(points, points_trans, separate_directions=True)
            loss = nn.MSELoss()
            return loss(pred, chamfer_loss)
        else:
            #pred = torch.reshape(pred, (pred.shape[0], 3, 3))
            trans = trnsfrm.Rotate(pred)

            points_t = points.transpose(2, 1)
            points_transformed2 = trans.transform_points(points_t)
            points_transformed2 = points_transformed2.transpose(2, 1)

            if loss_type == "chamfer":
                loss = get_cloud_chamfer_loss_tensor(points_transformed2, points_trans, separate_directions=False, reduction="mean")
            else:
                loss = torch.sum(torch.square(points_transformed2 - points_trans), dim=(1, 2))
                loss = loss.mean()
            #print("chamfer_loss: ", chamfer_loss)
            return loss
