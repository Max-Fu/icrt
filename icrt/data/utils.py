import torch 
from typing import Union, List, Tuple, Literal
import json 
import numpy as np
from scipy.spatial.transform import Rotation

def rot_mat_to_rot_6d(rot_mat : np.ndarray) -> np.ndarray: 
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :] # N, 2, 3
    return rot_6d.reshape(-1, 6) # N, 6

def rot_6d_to_rot_mat(rot_6d : np.ndarray) -> np.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat

def euler_to_rot_6d(euler : np.ndarray, format="XYZ") -> np.ndarray:
    """
    Convert euler angles to 6d representation
    euler: N, 3
    """
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_euler(rot_6d : np.ndarray, format="XYZ"):
    """
    Convert 6d representation to euler angles
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_euler(format, degrees=False)

def quat_to_rot_6d(quat : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to 6d representation
    quat: N, 4
    robomimic: 
    https://mujoco.readthedocs.io/en/2.2.1/programming.html#:~:text=To%20represent%203D%20orientations%20and,cos(a%2F2).
    To represent 3D orientations and rotations, MuJoCo uses unit quaternions - namely 4D unit vectors arranged as q = (w, x, y, z). 
    Here (x, y, z) is the rotation axis unit vector scaled by sin(a/2), where a is the rotation angle in radians, and w = cos(a/2). 
    Thus the quaternion corresponding to a null rotation is (1, 0, 0, 0). This is the default setting of all quaternions in MJCF.
    """
    assert format in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    if format == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)

def rot_6d_to_quat(rot_6d : np.ndarray, format : str = "wxyz") -> np.ndarray:
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    quat = Rotation.from_matrix(rot_mat).as_quat()
    if format == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def euler_to_quat(euler : np.ndarray, format_euler="XYZ", format_quat="wxyz") -> np.ndarray:
    """
    Convert euler angles to quaternion
    euler: N, 3
    """
    assert format_quat in ["wxyz", "xyzw"], "Invalid quaternion format, only support wxyz or xyzw"
    quat = Rotation.from_euler(format_euler, euler, degrees=False).as_quat()
    if format_quat == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def gram_schmidt(vectors : np.ndarray) -> np.ndarray: 
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows 

    vectors: batchsize, N, D 

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]
    
    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis

def combine_dicts(dlist,key):
    """
    Combine a list of dictionaries into a single dictionary.
    """
    d = {}
    for k in key:
        dk = [d[k] for d in dlist]
        d[k] = np.concatenate(dk, axis=0)
    return d

def calculate_delta_rot(euler_rot_start : np.ndarray, euler_rot_end : np.ndarray, format="XYZ") -> np.ndarray:
    """
    Calculate the delta rotation between two euler angles
    euler_rot_start: N, 3
    euler_rot_end: N, 3

    return: N, 3
    """
    r = Rotation.from_euler(format, euler_rot_start, degrees=False)
    r2 = Rotation.from_euler(format, euler_rot_end, degrees=False)
    delta_rot = r2 * r.inv()
    euler_rot = delta_rot.as_euler(format, degrees=False)
    return euler_rot

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def convert_multi_step(data : torch.Tensor, num_pred_steps: int):
    """Chunk data for predicting data `num_pred_steps` steps into the future.
    The resulting data have shape (batch, data.shape[-2] - (num_pred_steps - 1), num_pred_steps, action_dim)
    For example: chunk_data([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
            [a_4, a_5, a_5],
            [a_5, a_5, a_5],
        ]
    adapted from https://github.com/octo-models/octo/blob/7480a2a90160122b7a02459fc6f56ceefa501ebf/octo/model/components/action_heads.py#L59
    """
    assert (
        data.ndim == 2
    ), f"Expected data to have shape (seq length, action_dim), but got shape {data.shape}"
    window_size = data.shape[0]
    chunk_window_size = window_size

    curr_step = torch.arange(chunk_window_size, device=data.device)
    action_offset = torch.arange(num_pred_steps, device=data.device)
    chunk_indices = torch.minimum(curr_step[:, None] + action_offset[None, :], torch.tensor(chunk_window_size - 1))
    return data[chunk_indices]

def convert_delta_action(action, proprio):
    """
    Calculate the delta action given the action and proprioception
    Gripper action remains as absolute action
    action: S, T, action_dim
    proprio: S, T, proprio_dim
    """
    trans = action[:, :, :3].reshape(-1, 3)
    rot = action[:, :, 3:9].reshape(-1, 6)
    
    rot =  Rotation.from_matrix(rot_6d_to_rot_mat(rot))
    
    current_state = np.repeat(proprio[:, 0:1],action.shape[1],1)
    current_trans = current_state[:, :, :3].reshape(-1, 3)
    current_rot = current_state[:,:, 3:9]# S, T, 6
    current_rot =  Rotation.from_matrix(rot_6d_to_rot_mat(current_rot.reshape(-1, 6)))
    
    delta_rot = (current_rot.inv()*rot).as_matrix()
    delta_trans = np.einsum('ijk,ik->ij', current_rot.inv().as_matrix(),(trans-current_trans))

    delta_rot = rot_mat_to_rot_6d(delta_rot).reshape(-1,action.shape[1],6)
    delta_trans = delta_trans.reshape(-1,action.shape[1],3)
    
    if action.shape[-1] == proprio.shape[-1]:
        #no eos
        delta_action = np.concatenate([delta_trans, delta_rot, action[:,:,-1:]], axis=-1)
    else:
        #with eos
        delta_action = np.concatenate([delta_trans, delta_rot, action[:,:,-2:]], axis=-1)
    
    return delta_action
    
def convert_abs_action(action,proprio):
    '''
    Calculate the next state from the delta action and the current proprioception
    action: S, T, action_dim
    proprio: S, T, proprio_dim
    '''
    delta_trans = action[:, :, :3].reshape(-1, 3)
    delta_rot = action[:, :, 3:9].reshape(-1,6)
    delta_rot =  Rotation.from_matrix(rot_6d_to_rot_mat(delta_rot))
    
    current_state = np.repeat(proprio[:, 0:1],action.shape[1],1)
    current_trans = current_state[:, :, :3].reshape(-1, 3)
    current_rot = Rotation.from_matrix(rot_6d_to_rot_mat(current_state[:,:, 3:9].reshape(-1,6)))
    
    trans = np.einsum('ijk,ik->ij',current_rot.as_matrix(),delta_trans) + current_trans
    rot = (current_rot*delta_rot).as_matrix()
    
    rot = rot_mat_to_rot_6d(rot).reshape(-1,action.shape[1],6)
    trans = trans.reshape(-1,action.shape[1],3)
    
    if action.shape[-1] == proprio.shape[-1]:
        #no eos
        desired_mat = np.concatenate([trans, rot, action[:,:,-1:]], axis=-1)
    else:
        #with eos
        desired_mat = np.concatenate([trans, rot, action[:,:,-2:]], axis=-1)
    return desired_mat

def find_increasing_subsequences(arr : List[int]) -> List[Tuple[int, int]]:
    """
    4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3
    Find the all increasing subsequence in the order present in the dataset and return the values
    which should be [(4, 9), (1,9), (1,3)],
    
    args: 
        arr: List[int] - list of integers
    """
    subsequences = []
    start = arr[0] 
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] <= 0:
            subsequences.append((start, arr[i-1]))
            start = arr[i]
    subsequences.append((start, arr[-1]))
    return subsequences


def create_prompt_mask(eos_vector,num_steps):
    eos_position = sorted(np.where(eos_vector == 1)[0])
    if len(eos_position) == 0:
        return np.zeros_like(eos_vector), np.ones_like(eos_vector)
    #randomly select a position to mask
    if len(eos_position) >1:
        eos_position = eos_position[1:]
    mask_position = np.random.choice(eos_position)
    prompt_mask = np.zeros_like(eos_vector)
    prompt_mask[mask_position+1:] = 1 #if 1 then not prompt

    weight_mask = np.zeros_like(eos_vector)
    eos_position = eos_position[eos_position.index(mask_position):]
    eos_position.append(len(eos_vector))
    #the first one is the mask position
    if num_steps >= 1:
        for idx, eos in enumerate(eos_position[:-1]):
            end_pos = min(eos+num_steps+1, eos_position[idx+1])
            weight_mask[eos+1:end_pos] = 1
    else:
        #num_step is a ratio
        for idx, eos in enumerate(eos_position[:-1]):
            seq_len = eos_position[idx+1] - eos
            weight_steps = int(num_steps * seq_len)
            end_pos = eos+weight_steps+1
            weight_mask[eos+1:end_pos] = 1

    return prompt_mask, weight_mask

def scale_action(
        action : torch.Tensor, 
        stat : dict, 
        type : Literal["minmax", "standard"] = "standard"
) -> torch.Tensor: 
    """
    action: S, T, action_dim 
    stat: dictionary
    """
    # move stats to action device 
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = (action[..., :action_dim] - stat["min"]) / (stat["max"] - stat["min"])
    elif type == "standard":
        action[..., :action_dim] = (action[..., :action_dim] - stat["mean"]) / stat["std"]
    return action

def unscale_action(
        action : torch.Tensor, 
        stat : dict, 
        type : Literal["minmax", "standard"] = "standard"
) -> torch.Tensor: 
    """
    action: S, T, action_dim 
    stat: dictionary
    """
    # move stats to action device 
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = action[..., :action_dim] * (stat["max"] - stat["min"]) + stat["min"]
    elif type == "standard":
        action[..., :action_dim] = action[..., :action_dim] * stat["std"] + stat["mean"]
    return action
