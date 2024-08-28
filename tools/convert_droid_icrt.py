# python -m tools/convert_droid_icrt.py --droid_dir /home/lawrence/icrl/xembody_data_franka_r2d2 --out_dir data_output --visualize
# Known issue: if last byte of the image is zero, it has saving issues. 
# Currently, it is fixed by appending zeros to the image.
import os
import argparse 
import h5py
import numpy as np
from PIL import Image
from glob import glob
import json
from tqdm import tqdm
from collections import defaultdict

def init_hdf5_file(file, epi_idx, max_epi_len, resolution : tuple):
    
    resolution_bytes = resolution[0]*resolution[1]*3

    epi_name = f'real_episode_{epi_idx}'
    gp = file.create_group(epi_name)
    
    obs_gp = gp.create_group(f'observation')
    act_gp = gp.create_group(f'action')
    
    obs_gp.create_dataset(f'cartesian_position',(max_epi_len,6))
    obs_gp.create_dataset(f'gripper_position',(max_epi_len,1))
    obs_gp.create_dataset(f'joint_position',(max_epi_len,7))
    obs_gp.create_dataset(f'exterior_image_1_left',(max_epi_len),dtype=f'S{resolution_bytes}')#, compression="gzip")
    obs_gp.create_dataset(f'exterior_image_2_left',(max_epi_len),dtype=f'S{resolution_bytes}')#, compression="gzip")
    obs_gp.create_dataset(f'wrist_image_left',(max_epi_len),dtype=f'S{resolution_bytes}')#, compression="gzip")
    
    act_gp.create_dataset(f'cartesian_position',(max_epi_len,6))
    act_gp.create_dataset(f'gripper_position',(max_epi_len,1))
    act_gp.create_dataset(f'joint_position',(max_epi_len,7))
    act_gp.create_dataset(f'cartesian_velocity',(max_epi_len,6))
    act_gp.create_dataset(f'gripper_velocity',(max_epi_len,1))
    act_gp.create_dataset(f'joint_velocity',(max_epi_len,7))
    
    gp.create_dataset(f'language_instruction',(1,),dtype='S200') # needs to decode when read, maximum 500 characters
    gp.create_dataset(f'language_instruction_2',(1,),dtype='S200') # needs to decode when read, maximum 500 characters
    gp.create_dataset(f'language_instruction_3',(1,),dtype='S200') # needs to decode when read, maximum 500 characters
    gp.create_dataset(f'language_embedding',(1,512))
    gp.create_dataset(f'language_embedding_2',(1,512))
    gp.create_dataset(f'language_embedding_3',(1,512))
    
    other_keys = ['is_first', 'is_last', 'is_terminal', 'reward', 'discount']
    for key in other_keys:
        gp.create_dataset(key,(max_epi_len))

    return epi_name, other_keys

def step_to_dict(step):
    # recursively convert all tensors to numpy arrays
    step_dict = {}
    for k, v in step.items():
        if isinstance(v, dict):
            step_dict[k] = step_to_dict(v)
        else:
            step_dict[k] = v.numpy()
            if isinstance(step_dict[k], bytes):
                step_dict[k] = step_dict[k].decode('utf-8')
    return step_dict

# choose the dataset path in the dropdown on the right and rerun this cell
# to see multiple samples

def key_mapping(key):
    if key == "hand_camera_left_image":
        return "wrist_image_left"
    elif key == "varied_camera_1_left_image":
        return "exterior_image_1_left"
    elif key == "varied_camera_2_left_image":
        return "exterior_image_2_left"
    else:
        return key

def as_gif(images, path='temp.gif'):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000//30, loop=0)
    gif_bytes = open(path,'rb').read()
    return gif_bytes

if __name__ == "__main__":

    """
    structure of droid data collection
    - /home/lawrence/icrl/xembody_data_franka_r2d2
        - 2024-01-24-tiger-background (task)
            - Wed_Jan_24_19:54:24_2024 (trajectory)
                - trajectory_im128.h5
                    - 'action'
                        - cartesian_position, gripper_position, joint_position, cartesian_velocity, gripper_velocity, joint_velocity
                    - 'observation'
                        - 'camera', 
                            - 'hand_camera_left_image', -> wrist_image_left
                            - 'varied_camera_1_left_image', -> exterior_image_1_left
                            - 'varied_camera_2_left_image' -> exterior_image_2_left
                        - 'robot_state',
                            - 'cartesian_position', 'gripper_position', 'joint_positions', 'joint_velocities', 
    """
    parser = argparse.ArgumentParser(description='Convert dataset to hdf5 format')
    parser.add_argument('--droid_dir', type=str, default='/home/lawrence/icrl/xembody_data_franka_r2d2', help='Dataset to convert')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize the dataset')
    parser.add_argument('--resolution', type=int, nargs="+", default=(180, 320), help='Resolution of the images')
    args = parser.parse_args()

    # resolution
    resolution = args.resolution 
    if len(resolution) == 1:
        resolution = (resolution[0], resolution[0])
    image_keys = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']

    os.makedirs(args.out_dir, exist_ok=True)

    # tasks = os.listdir(args.droid_dir)
    tasks=[d for d in os.listdir(args.droid_dir) if os.path.isdir(os.path.join(args.droid_dir, d))]
    language_instructions = set()
    language_json_path = os.path.join(args.out_dir, "language_instructions.json")

    h5_file = h5py.File(args.out_dir+f'/r2d2.hdf5', 'w')

    hdf5_keys = []
    epi_len_mapping_json = {}
    verb_to_episode = defaultdict(list)

    for task in tasks:
        task_dir = os.path.join(args.droid_dir, task)
        trajectories = os.listdir(task_dir)
        print("Processing task dir: ", task_dir)
        print("Number of trajectories: ", len(trajectories))
        task_epi_idx = 0
        for trajectory in tqdm(trajectories):
            # find the trajectory file
            traj_h5 = glob(os.path.join(task_dir, trajectory, "trajectory_im320_180.h5"))[0]
            # traj_h5 = os.path.join(task_dir, trajectory, 'trajectory_im320_180.h5')
            with h5py.File(traj_h5, 'r') as f:
                episode_length = f["observation/robot_state/cartesian_position"].shape[0]
                epi_key, other_keys = init_hdf5_file(h5_file, task+f'_{task_epi_idx}', episode_length, resolution)

                # save the keys for the hdf5 file
                hdf5_keys.append(epi_key)
                epi_len_mapping_json[epi_key] = episode_length
                verb_to_episode[task].append(epi_key)

                # observations 
                for obs_key in f["observation/camera/image"].keys():
                    # if obs keys in obs_gp, save to hdf5
                    new_obs_key = key_mapping(obs_key)
                    if new_obs_key in h5_file[epi_key]['observation'].keys():
                        for idx in range(episode_length):
                            # dtype('uint8')
                            h5_file[epi_key]['observation'][new_obs_key][idx] = f["observation/camera/image"][obs_key][:][idx].tobytes()
                            # # debug only 
                            # data_shape = np.prod(f["observation/camera/image"][obs_key][:][epi_idx].shape)
                            # encoded_image = f["observation/camera/image"][obs_key][:][epi_idx].tobytes()
                            # decoded_image = np.frombuffer(encoded_image, dtype='uint8')
                            # decoded_image_dataset = np.frombuffer(h5_file[epi_key]['observation'][new_obs_key][epi_idx], dtype='uint8')
                            # if data_shape != 128*128*3 or decoded_image.shape[0] != 128*128*3 or decoded_image_dataset.shape[0] != 128*128*3:
                            #     import pdb; pdb.set_trace()
                            #     print("decoded_image_dataset.shape[0]: ", decoded_image_dataset.shape[0])
                            #     print("decoded_image.shape[0]: ", decoded_image.shape[0])
                            #     print("len encoded_image: ", len(encoded_image))
                            #     print("len encoded image dataset: ", len(h5_file[epi_key]['observation'][new_obs_key][epi_idx]))
                            # # end debug only 
                
                # states 
                for state_key in f["observation/robot_state"].keys():
                    if state_key in h5_file[epi_key]['observation'].keys():
                        if f["observation/robot_state"][state_key][:].ndim == 1:
                            # unsqueeze the last dimension 
                            state_data = f["observation/robot_state"][state_key][:][:, None]
                        else:
                            state_data = f["observation/robot_state"][state_key][:]
                        h5_file[epi_key]['observation'][state_key][:] = state_data

                # actions
                for act_key in f["action"].keys():
                    if act_key in h5_file[epi_key]['action'].keys():
                        h5_file[epi_key]['action'][act_key][:] = f["action"][act_key][:]
                
                # language keys 
                h5_file[epi_key]["language_instruction"][0] = task
                # we don't deal with other keys for now

            task_epi_idx += 1

    
    h5_file.close()

    # save the json files to the output directory
    with open(os.path.join(args.out_dir, "epi_len_mapping.json"), 'w') as f:
        json.dump(epi_len_mapping_json, f)
    
    with open(os.path.join(args.out_dir, "hdf5_keys.json"), 'w') as f:
        json.dump(hdf5_keys, f)

    with open(os.path.join(args.out_dir, "verb_to_episode.json"), 'w') as f:
        json.dump(verb_to_episode, f)

    if args.visualize:
        # visualize the dataset 
        with h5py.File(args.out_dir+f'/r2d2.hdf5', 'r') as f:
            # visualize the observations 
            for epi_idx in f.keys():
                print(f"Episode {epi_idx}")
                for obs_key in f[f'{epi_idx}/observation'].keys():
                    episode_length = f[f'{epi_idx}/observation/wrist_image_left'].shape[0]
                    if obs_key in image_keys:
                        print(f"Visualizing {obs_key}")
                        images = []
                        for i in range(episode_length):
                            img = np.frombuffer(f[f'{epi_idx}/observation/{obs_key}'][i],dtype='uint8')
                            total_bytes = resolution[0]*resolution[1]*3
                            if img.shape[0] != total_bytes:
                                # append zeros to the image
                                img = np.append(img, np.zeros(total_bytes - img.shape[0])).astype('uint8')
                            img = img.reshape(resolution[0], resolution[1] ,3)
                            images.append(Image.fromarray(img))
                            
                        as_gif(images, path=args.out_dir+f"/{obs_key}_epi_{epi_idx}.gif")