import argparse
import json
import numpy as np
import time
import tyro 
import yaml
import os
import sys
import traceback

from collections import OrderedDict

from pathlib import Path
from typing import Union

import torch
import torch.backends.cudnn as cudnn

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import robomimic.utils.tensor_utils as TensorUtils
import torch.nn as nn
from torchvision import transforms

import timm
from icrt.data.dataset import SequenceDataset
import icrt.util.misc as misc
from icrt.util.args import ExperimentConfig
from icrt.util.model_constructor import model_constructor
from icrt.data.utils import rot_6d_to_euler, quat_to_rot_6d, euler_to_rot_6d


import kinpy

Max_Gripper_Width = 0.085
chain = kinpy.build_chain_from_urdf(open("assets/franka/panda.urdf").read())


class ICRLRolloutPolicy(RolloutPolicy):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, dataset, action_exec_horizon, max_prompt_len=0, preprocess=None,
                 image_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                 proprio_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos','robot0_joint_pos'],
                 rot_6d=False,
                 ):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        super().__init__(policy)

        self.preprocess = preprocess
        
        if self.preprocess is not None:
            self.preprocess = transforms.Compose([t for t in self.preprocess.transforms if not isinstance(t, transforms.ToTensor)])
        else:
            print("warning: vision transforms are not defined. Using default transforms.")
            self.preprocess = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC, antialias='warn'), # kept consistent with default
                transforms.CenterCrop(size=224),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
            
        self.image_keys = image_keys
        self.proprio_keys = proprio_keys
        self.dataset = dataset
        self.action_exec_horizon = action_exec_horizon
        self.max_prompt_len = max_prompt_len
        self.rot_6d = rot_6d
        self.obs_history = []
        self.act_history = []

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.reset(self.action_exec_horizon)
        print(self.max_prompt_len)
        if self.max_prompt_len != 0:
            demo_idx = np.random.randint(len(self.dataset))
            demo_sequences = self.dataset.__getitem__(demo_idx)
            self.last_observation = None
            sequences = self._prepare_prompt(demo_sequences)
            
            # make sure that the sequence ends on an eos token 
            eos = sequences["action"].squeeze()[..., 0, -1]
            last_eos_idx = torch.where(eos)[0][-1]
            
            # cut the observation, proprio, action to the last eos index position 
            for k in sequences:
                sequences[k] = sequences[k][:, :last_eos_idx+1]
            
            if self.max_prompt_len > 0:
                for k in sequences:
                    sequences[k] = sequences[k][:, -self.max_prompt_len :].contiguous()
            self.policy.prompt(sequences)
        


    def _prepare_observation(self, ob, add_eos=True):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        meta = {}
        meta["observation"] = np.zeros((len(self.image_keys), ob[self.image_keys[0]].shape[0], ob[self.image_keys[0]].shape[-3], 
                                        ob[self.image_keys[0]].shape[-2], ob[self.image_keys[0]].shape[-1]))
        for i in range(len(self.image_keys)):
            meta["observation"][i,:,:,:,:] = ob[self.image_keys[i]]
        meta["observation"] = np.transpose(meta["observation"], (1, 0, 2, 3, 4)) # T, N, 3, H, W
        
        dtype = ob[self.image_keys[0]].dtype
        norm = 1.0
        if dtype == np.uint8:
            norm = 255.0
        meta["observation"] = meta["observation"] / norm
        
        # proprio
        gripper_width = ob['robot0_gripper_qpos'][:,0] - ob['robot0_gripper_qpos'][:,1]
        ob['robot0_gripper_qpos'] = (1 - gripper_width/Max_Gripper_Width)
        
        for i in range(len(ob['robot0_gripper_qpos'])):
            #calculate ee using fk and update ob
            joint_poses = ob['robot0_joint_pos'][i]
            th = dict(zip(chain.get_joint_parameter_names(), joint_poses))
            ee_ = chain.forward_kinematics(th)
            ee_1 = ee_['panda_leftfinger']
            ee_2 = ee_['panda_rightfinger']
            ee_pos_mid = (ee_1.pos + ee_2.pos)/2
            ee_quat_mid = (ee_1.rot + ee_2.rot)/2
            ob['robot0_eef_pos'][i] = ee_pos_mid
            ob['robot0_eef_quat'][i] = ee_quat_mid
            
        meta["proprio"] = np.concatenate((ob["robot0_eef_pos"], ob['robot0_eef_quat'], ob['robot0_gripper_qpos'][:, None]), axis=-1) 
        
        ob = meta

        if self.rot_6d:
            ob = self._update_rot_6d(ob)

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)    
        
        obs = torch.reshape(ob["observation"], (-1, *ob["observation"].shape[-3:]))
        obs = self.preprocess(obs)
        ob["observation"] = obs.view(*ob["observation"].shape[:-2], 224, 224).float()        
            
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        ob = TensorUtils.to_device(ob, device)
        ob = TensorUtils.to_float(ob)
        
        return ob

    def _update_rot_6d(self, ob):
        """
        update proprio and action to 6d rotation
        """
        ob["proprio"] = np.concatenate([ob["proprio"][:, :3], quat_to_rot_6d(ob["proprio"][:, 3:7]), ob["proprio"][:, 7:]], axis=-1)
        if ob["action"] is not None:
            ob["action"] = np.concatenate([ob["action"][:, :3], euler_to_rot_6d(ob["action"][:, 3:6]), ob["action"][:, 6:]], axis=-1)
            print("converted 6d action (should be same as predicted action): ", ob["action"])
        return ob

    def _prepare_prompt(self, ob : dict):
        # prepare the sequence loaded from the dataset
        if self.rot_6d:
            ob = self._update_rot_6d(ob)

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
      
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        ob = TensorUtils.to_device(ob, device)
        ob = TensorUtils.to_float(ob)
      
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
            goal (dict): goal observation
        """
        for k in ob:
            ob[k] = ob[k][None]
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        
        try:
            self.obs_history.append(ob['proprio'].detach().cpu().numpy())
            self.act_history.append(ob['action'].detach().cpu().numpy())
        except:
            pass
        
        ac = self.policy.get_action(ob)
        # print("predicted action: ", ac)
        action = TensorUtils.to_numpy(ac)

        if self.rot_6d:
            # convert action from 6d to euler
            action = np.concatenate([action[:3], rot_6d_to_euler(action[3:9][None, :]).flatten(), action[9:]], axis=-1)
            print("converted euler action: ", action)

        return action

    

def run(config, env_meta, task, args, rollout_model):
    """
    Run the policy from model in the environment.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    
    # print(config)
    log_dir = os.path.join(args.logging_cfg.log_dir, task+ "_logs")
    os.makedirs(log_dir, exist_ok=True)
    video_dir = os.path.join(args.logging_cfg.log_dir, task+ "_videos")
    os.makedirs(video_dir, exist_ok=True)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    # create environments for validation runs
    env_names = [env_meta["env_name"]]



    for env_name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name, 
            render=False, 
            render_offscreen=config.experiment.render_video,
            use_image_obs=True,
            use_depth_obs=False,
        )
        envs[env.name] = env
        print(envs[env.name])


    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # print("\n============= Model Summary =============")
    # print(model)  # print model summary
    # print("")


    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    
    num_episodes = config.experiment.rollout.n
    epoch = 1
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
    )
    
    # summarize results from rollouts to tensorboard and terminal
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]
        for k, v in rollout_logs.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
            else:
                data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

        print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
        print('Env: {}'.format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))


    # terminate logging
    data_logger.close()


def main(
    train_yaml_path : Union[str, Path],
    checkpoint_path : Union[str, Path],
    train_val : str = "val",
    data_config: str = "config/robomimic_eval.json",
    n_episodes: int = 10,
    max_prompt_len: int = 0,
    action_exec_horizon: int = 1,
    task_name : str = "all", 
    ):
    
    """Plotting the predicted actions vs the ground truth actions

    Args: 
        train_yaml_path: str, path to the yaml file containing the training configuration
        checkpoint_path: str, path to the checkpoint to load
        demo_index: int, index of the demo to plot
        train_val: str, "train" or "val", whether to use the train or val set
        task_name: str, one of ["all", "square", "can", "lift"]
    """
    # we evaluate using the following pretrained_path
    args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader) 

    data_json = args.dataset_cfg.dataset_json
    with open(data_json, 'r') as f:
        data_json = json.load(f)
    dataset_path = data_json["train"]["dataset_path"]

    model_output_dir = args.logging_cfg.output_dir
    # creating the output directory and logging directory for test
    args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, "test_output")
    args.logging_cfg.log_dir = args.logging_cfg.output_dir
    output_dir = args.logging_cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Logging results to ", output_dir)
    
    

    assert os.path.exists(checkpoint_path), f"Checkpoint path does not exist: {checkpoint_path}"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Loading data config
    data_cfg = json.load(open(args.dataset_cfg.dataset_json, 'r'))

    # make sure the number of cameras is correct 
    rgb_observations = data_cfg["observation"]["modalities"]["obs"]["rgb"]
    assert len(rgb_observations) == args.shared_cfg.num_cameras, "Number of cameras must match the number of rgb observations"

    model = model_constructor(
        model_config=args.model_cfg, 
        shared_config=args.shared_cfg,
        train=args.train,
    )

    timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
    vision_transform = timm.data.create_transform(**timm_data_cfg)
    
    print("vision transform: ", vision_transform)

    model.to(device)
    

    
    total, trainable = model.get_total_parameters(), model.get_trainable_parameters()
    print("trainable: ", trainable)
    print("Total params: ", total)
    print("percentage trainable: ", trainable / total)
    
    # loading pretrained checkpoint
    print("loading pretrained model from: ", checkpoint_path)
    misc.load_model(model, checkpoint_path)
    
    if train_val == "train":
        dataset = SequenceDataset(
            dataset_config=args.dataset_cfg,
            shared_config=args.shared_cfg,
            vision_transform=vision_transform,
            split_file=os.path.join(model_output_dir, "train_split.json")
        )
    else: 
        dataset = SequenceDataset(
            dataset_config=args.dataset_cfg,
            shared_config=args.shared_cfg,
            vision_transform=vision_transform,
            split_file=os.path.join(model_output_dir, "val_split.json")
        )

    # we evaluate the model here 
    model.eval()
    
    

    # reset the rotation back
    assert dataset.rot_6d == args.shared_cfg.rot_6d 

    rollout_model = ICRLRolloutPolicy(model, dataset, action_exec_horizon, max_prompt_len, preprocess=vision_transform, rot_6d=args.shared_cfg.rot_6d)
    
    
    
    ext_cfg = json.load(open(data_config, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.unlocked():
        config.update(ext_cfg)
        

    config.train.data = [dataset_path]

    if args.logging_cfg.log_name is not None:
        config.experiment.name = args.logging_cfg.log_name
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # override config from args
    config.train.cuda = args.device == 'cuda'
    config.train.seq_length = args.shared_cfg.seq_length
    config.train.hdf5_cache_mode = None
    config.train.output_dir = args.logging_cfg.output_dir
    config.train.seed = seed
    config.experiment.rollout.n = n_episodes

    print("\n============= Loaded Environment Metadata =============")
    with open('/shared/projects/icrl/data/robomimic_datasets/env_attr_square_lift_can.json') as json_data:
        env_metas = json.load(json_data)
    tasks = list(env_metas.keys())
    if task_name == "all":
        tasks = tasks
    else:
        assert task_name in tasks, f"Task {task_name} not found in {tasks}"
        tasks = [task_name]

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

  
    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    for task in tasks:
        print("\n============= Running Task {} =============".format(task))
        # catch error during training and print it
        try:
            run(config, env_metas[task], 'robomimic_'+task, args, rollout_model)
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)
        



if __name__ == "__main__":
    tyro.extras.set_accent_color("yellow")
    tyro.cli(main)