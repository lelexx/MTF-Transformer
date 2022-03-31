from os import device_encoding
import torch
import numpy as np
import math 
import sys
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch.nn.functional as F
#from common.svd.batch_svd import batch_svd
def eval_metrc(cfg, predicted, target):
    '''
    predicted:(B, T, J, C)
    '''
    B, T, J, C = predicted.shape
    if cfg.TEST.METRIC == 'mpjpe':
        eval_loss = mpjpe(predicted, target)
    elif cfg.TEST.METRIC == 'p_mpjpe':
        predicted = predicted.view(B*T, J, C)
        target = target.view(B*T, J, C)
        eval_loss = p_mpjpe(cfg, predicted, target)
    elif cfg.TEST.METRIC == 'n_mpjpe':
        eval_loss = n_mpjpe(predicted, target)

    return eval_loss


def get_mat_torch(x, y): 
    z = torch.cross(y, x)
    y = torch.cross(x, z)
    x = torch.cross(z, y)
    mat = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)
    return mat 


def get_poses_torch(joints): 

    # input joints expected to be (N, 17, 3)
    # the parent and child link of some joint 
    parents, children = [], []
    # r knee 
    xp = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yp = joints[:, 2] - joints[:, 1]
    xc = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yc = joints[:, 3] - joints[:, 2]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # r hip 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = torch.cross(joints[:, 1] - joints[:, 2], joints[:, 3] - joints[:, 2])
    yc = joints[:, 2] - joints[:, 1]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # l hip 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yc = joints[:, 5] - joints[:, 4]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # l knee 
    xp = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yp = joints[:, 5] - joints[:, 4]
    xc = torch.cross(joints[:, 4] - joints[:, 5], joints[:, 6] - joints[:, 5])
    yc = joints[:, 6] - joints[:, 5]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # re 
    xp = joints[:, 15] - joints[:, 14]
    yp = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    xc = joints[:, 16] - joints[:, 15]
    yc = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # rs 
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = joints[:, 15] - joints[:, 14]
    yc = torch.cross(joints[:, 14] - joints[:, 15], joints[:, 16] - joints[:, 15])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # ls 
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = joints[:, 11] - joints[:, 12]
    yc = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # le 
    xp = joints[:, 11] - joints[:, 12]
    yp = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    xc = joints[:, 12] - joints[:, 13]
    yc = torch.cross(joints[:, 13] - joints[:, 12], joints[:, 11] - joints[:, 12])
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # thorax 
    xp = joints[:, 1] - joints[:, 4]
    yp = joints[:, 0] - joints[:, 7]
    xc = joints[:, 14] - joints[:, 11]
    yc = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    # pelvis 
    
    xc = joints[:, 1] - joints[:, 4]
    yc = ((joints[:, 1] + joints[:, 4]) / 2 - joints[:, 0]) * 100.0 
    mask = torch.nonzero((((joints[:, 1] + joints[:, 4]) / 2 - joints[:, 0]) * 100.0).sum(dim=-1) < 10.0)
    if len(mask) > 0: 
        mask = mask.squeeze()
        yc[mask] = joints[mask, 0] - joints[mask, 7]

    c = get_mat_torch(xc, yc)
    children.append(c.unsqueeze(1))

    # neck 8
    xp = joints[:, 14] - joints[:, 11]
    yp = joints[:, 7] - (joints[:, 11] + joints[:, 14]) / 2
    xc = torch.cross(joints[:, 8] - joints[:, 10], joints[:, 9] - joints[:, 10])
    yc = (joints[:, 11] + joints[:, 14]) / 2 - joints[:, 10]
    p = get_mat_torch(xp, yp)
    c = get_mat_torch(xc, yc)
    parents.append(p.unsqueeze(1)); children.append(c.unsqueeze(1))
    
    #[r knee, r hip, l hip, l knee, re, rs, ls, le, thorax, pelvis, neck 8]
    # concat -> out shape: (N, 11, 3, 3)
    N = len(parents)
    parents = torch.cat(parents, dim=1)
    children = torch.cat(children, dim=1)
    
    
    # normalize 
    parents = parents / (parents.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8)
    children = children / (children.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8)

 
    return parents, children
def align_numpy(source, target):
    '''
    Args:
        source : (B, J, C)
        target : (B, J, C)
        vis:     (B, J, 1)
    '''
    if type(source) == torch.Tensor:
        source = source.numpy()
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(source, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
    
    H = np.matmul(X0.transpose(0, 2, 1), Y0) 
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    return torch.from_numpy(R)
    


def align_target_numpy(cfg, source, target):
    '''
    Args:
        source : (B, T, J, C, N)
        target : (B, T, J, C, N)
    '''
    B, T, J, C, N = source.shape
    source = source.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    target = target.permute(0, 1, 4, 2, 3).contiguous().view(B*T*N, J, C) 
    if type(source) == torch.Tensor:
        source = source.numpy()
        target = target.numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(source, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
    
    H = np.matmul(X0.transpose(0, 2, 1), Y0) 
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    if cfg.TEST.TRJ_ALIGN_R:
        source = np.matmul(source, R)
    if cfg.TEST.TRJ_ALIGN_S:
        source = a * source
    if cfg.TEST.TRJ_ALIGN_T:
        source = source + t
    
    source = torch.from_numpy(source)
    source = source.view(B, T, N, J, C).permute(0, 1, 3, 4, 2)  #B, T, J, C, N

    return source

def align_torch(source, target):
    '''
    Args:
        source : (B, J, C)
        target : (B, J, C)
        vis:     (B, J, 1)
    '''
    
    device = source.device
    assert len(source.shape) == 3

    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(source, dim=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
     
    H = torch.matmul(X0.permute(0, 2, 1), Y0) 
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    return R
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    return torch.from_numpy(R).to(device)
def test_multi_view_aug(pred, vis):
    '''
    Args:
        pred:(B, T, J, C, N) T = 1
        vis: (B, T, J, C, N) T >= 1
    '''
    B, T, J, C, N = pred.shape
    
    pad = T // 2
    if vis is not None:
        vis = vis[:,pad:pad+1] #(B, 1, J, 1, N)
    else:
        vis = torch.ones(B, 1, J, 1, N)
    att = vis.view(B, T, J, 1, 1, N).repeat(1, 1, 1, 1, N, 1)

    if N == 1:
        return pred
    else:
        final_out = torch.zeros(B*T, J, C, N, N).float()
        pred = pred.view(B*T, J, C, N)
        for view_id in range(N):
            final_out[:,:,:,view_id, view_id] = pred[:,:,:,view_id]
            
        for view_list in itertools.combinations(list(range(N)), 2):
            view_1_id = view_list[0]
            view_2_id = view_list[1]

            R = align_numpy(source=pred[:,:,:,view_2_id], target=pred[:,:,:,view_1_id])

            final_out[:,:,:,view_1_id, view_2_id] = torch.matmul(pred[:,:,:,view_2_id], R)
            final_out[:,:,:,view_2_id, view_1_id] = torch.matmul(pred[:,:,:,view_1_id], R.permute(0, 2, 1)) 

        att = F.softmax(att, dim=-1).float() #(B, T, J, C, N, N)  
        final_out = final_out.view(B, T, J, C, N, N) * att
        final_out = torch.sum(final_out, dim = -1)
        
        return final_out

        
def mpjpe(predicted, target):
    assert predicted.shape == target.shape   
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


def p_mpjpe(cfg, predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    if type(predicted) == torch.Tensor:
        predicted = predicted.numpy()
    if type(target) == torch.Tensor:
        target = target.numpy()
        
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    if cfg.TEST.METRIC_ALIGN_R:
        predicted = np.matmul(predicted, R)
    if cfg.TEST.METRIC_ALIGN_S:
        predicted = a * predicted
    if cfg.TEST.METRIC_ALIGN_T:
        predicted = predicted + t


    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

def mv_mpjpe(pred, gt, mask):
    """
    pred:(B, T, N, J, C)
    gt: (B, T, N, J, C)
    mask: (B, N, N)
    """

    B, T, N, J, C = pred.shape
    
    loss = 0
    pad = T // 2
    
    num = 0
    for b in range(mask.shape[0]):
        for view_pair in itertools.combinations(list(range(mask.shape[-1])), 2):
            view_1_id = view_pair[0]
            view_2_id = view_pair[1]
            m_1 = mask[b, view_1_id]
            m_2 = mask[b, view_2_id]
            if torch.equal(m_1, m_2):
                R = align_numpy(source=gt[b:b+1, 0, view_2_id].cpu(), target=gt[b:b+1, 0, view_1_id].cpu())
                tmp = torch.einsum('btjc,bck->btjk', pred[b:b+1,:,view_2_id], R.to(pred.device))

                loss = loss + mpjpe(tmp, pred[b:b+1,:,view_1_id])
                num += 1

    return loss / (num + 1e-9)

def get_rotation(target):
    #B, T, J, C, N

    device = target.device
    target = target.permute(0, 1, 4, 2, 3) #(B, T, N, J, C)
    B, T, N, J, C = target.shape
    predicted = target.view(B, T, 1, N, J, C).repeat(1, 1, N, 1, 1, 1).view(-1, J, C)
    target = target.view(B, T,N,1, J, C).repeat(1, 1, 1, N, 1, 1).view(-1, J, C)
    
    if type(predicted) == torch.Tensor:
        predicted = predicted.detach().cpu().numpy()
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
        
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    
    R = torch.from_numpy(R).float().to(device)
    R = R.view(B, T, N, N, 3, 3)
    R = R.permute(0, 5, 4, 1, 2, 3) #(B,  3, 3,T, N, N,)

    return R


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted**2, dim=-1, keepdim=True), dim=-2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=-1, keepdim=True), dim=-2, keepdim=True)
    scale = norm_target / norm_predicted
    
    return mpjpe(scale * predicted, target)



def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))
