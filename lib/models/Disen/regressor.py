import numpy as np
import torch
import torch.nn as nn
from ..smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class GlobalRegressor(nn.Module):
    def __init__(self, dim, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(GlobalRegressor, self).__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        #self.fc2 = nn.Linear(1024, 1024)
        #self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, is_train=False, J_regressor=None):
        B, T = x.shape[:2]
        
        seq_len = x.shape[1]
        x = x.reshape(-1, x.size(-1))
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        else : 
            init_pose = init_pose.reshape(-1, init_pose.size(-1))
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        else : 
            init_shape = init_shape.reshape(-1, init_shape.size(-1))
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        else : 
            init_cam = init_cam.reshape(-1, init_cam.size(-1))

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            #xc = self.fc2(xc)
            #xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]

        scores = None
        for s in output:
            s['theta'] = s['theta'].reshape(B, T, -1)
            s['verts'] = s['verts'].reshape(B, T, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, T, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, T, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, T, -1, 3, 3)
            s['scores'] = scores

        return output
    
def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

class NewLocalRegressor(nn.Module):
    def __init__(self, dim, smpl_mean_params=SMPL_MEAN_PARAMS, hidden_dim=1024, drop=0.5):
        super(NewLocalRegressor, self).__init__()
        npose = 24 * 6
        self.fc1 = nn.Linear(dim + 10, hidden_dim)
        self.fc2 = nn.Linear(dim + npose, hidden_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim * 2 + 3, 3)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        self.decpose = nn.Linear(hidden_dim, 144)


    def forward(self, x, init_pose, init_shape, init_cam, is_train=False, J_regressor=None):
        pred_pose = init_pose.detach()
        pred_shape = init_shape.detach()
        pred_cam = init_cam.detach()
        
        xc_shape_cam = torch.cat([x, pred_shape], -1)

        xc_shape_cam = self.fc1(xc_shape_cam)
        xc_shape_cam = self.drop1(xc_shape_cam)
       
        for _ in range(3):
            xc_pose_cam = torch.cat([x, pred_pose], dim=-1)     
            xc_pose_cam = self.fc2(xc_pose_cam)                               
            xc_pose_cam = self.drop2(xc_pose_cam)
            pred_pose = self.decpose(xc_pose_cam) + pred_pose      

        pred_shape = self.decshape(xc_shape_cam) + pred_shape  
        pred_cam = self.deccam(torch.cat([xc_pose_cam, xc_shape_cam, pred_cam], -1)) + pred_cam

        pred_pose = pred_pose.reshape(-1, 144)
        pred_shape = pred_shape.reshape(-1, 10)
        pred_cam = pred_cam.reshape(-1, 3)
        batch_size = pred_pose.shape[0]

        out_put = self.get_output(pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor)
        return out_put

    def get_output(self, pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor):
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output