import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import yaml
import numpy as np
import timm
from typing import Union, Optional, List
from pathlib import Path
import PIL

from icrt.util.args import ExperimentConfig
import icrt.util.misc as misc
from icrt.util.model_constructor import model_constructor
from icrt.data.utils import rot_6d_to_euler, quat_to_rot_6d, euler_to_rot_6d
from icrt.data.utils import convert_delta_action

class ICRTWrapper(nn.Module):
    def __init__(
        self, 
        train_yaml_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        vision_encoder_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        # loading experiment config 
        args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        self.args = args
        self.device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.shared_cfg.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True
        
        # start model construction
        if vision_encoder_path is not None:
            args.model_cfg.vision_encoder_cfg.vision_encoder = vision_encoder_path
        else:
            print("Vision encoder is loaded from the model checkpoint! ")

        model = model_constructor(
            model_config=args.model_cfg, 
            shared_config=args.shared_cfg,
            train=False,
        )

        # obtain vision transforms 
        timm_data_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
        self.preprocess = timm.data.create_transform(**timm_data_cfg)
        self.mean, self.std = timm_data_cfg["mean"], timm_data_cfg["std"]
        
        print("vision transform: ", self.preprocess)
        model.to(self.device)

        total, trainable = model.get_total_parameters(), model.get_trainable_parameters()
        print("trainable: ", trainable)
        print("Total params: ", total)
        print("percentage trainable: ", trainable / total)
        
        # loading pretrained checkpoint
        print("loading pretrained model from: ", checkpoint_path)
        misc.load_model(model, checkpoint_path)
        model.eval()

        self.model = model

        if self.preprocess is not None:
            self.preprocess = transforms.Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, transforms.ColorJitter)]
            )
        else:
            print("warning: vision transforms are not defined. Using default transforms.")
            self.preprocess = transforms.Compose([
                transforms.Resize(size=248, max_size=None, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        self.reset()

    def reset(self):
        self.model.reset(self.model.num_pred_steps)

    def prompt(
        self,
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image]], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image]], 
        proprio : Union[np.ndarray], 
        action : Optional[np.ndarray] = None,
    ):
        """
        Prompt the model with a demo

        Args:
            side_image (PIL.Image.Image or List[PIL.Image.Image]): side camera image
            wrist_image (PIL.Image.Image or List[PIL.Image.Image]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            action (np.ndarray): action information
        """
        demo_sequence = self.prepare_observations(side_image, wrist_image, proprio, action)
        self.reset()
        for k, v in demo_sequence.items():
            demo_sequence[k] = v.to(self.device, non_blocking=True)
        self.model.prompt(demo_sequence)

    def prepare_observations(
        self, 
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image]], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image]], 
        proprio : Union[np.ndarray], 
        action : Optional[np.ndarray] = None
    ):
        """
        assume the observation is a dictionary with the following keys:
        "observation", "proprio", "action"

        proprio in xyzXYZGripper format
        action in xyzXYZGripper format
        """
        if isinstance(side_image, PIL.Image.Image):
            side_image = [side_image]
        if isinstance(wrist_image, PIL.Image.Image):
            wrist_image = [wrist_image]
        
        # find the length of the sequence 
        seq_len = len(side_image)
        assert len(side_image) == len(wrist_image) == len(proprio), "Length of the sequence must be the same"
        if action is not None:
            assert len(action) + 1 == seq_len or len(action) == seq_len, "Length of the action sequence must be one less than the observation sequence"
        assert proprio.shape[0] == seq_len, "Length of proprio sequence must match the length of the observation sequence"
        
        # construct input dictionary 
        side_image = torch.cat([self.preprocess(img)[None] for img in side_image], dim=0)
        wrist_image = torch.cat([self.preprocess(img)[None] for img in wrist_image], dim=0) 
        # interleave the images 
        image_vec = torch.stack([side_image, wrist_image], dim=1)[None] # stack along a new axis (seq_len, 2, 3, H, W), then add batch dim

        # process proprio 
        rot = proprio[:, 3:-1]
        if self.args.shared_cfg.rot_6d:
            rot = euler_to_rot_6d(rot)
        proprio = np.concatenate([proprio[:, :3], rot, proprio[:, -1:]], axis=-1) 
        proprio = torch.tensor(proprio)[None].float()

        # process action 
        if action is not None:
            rot = action[:, 3:-1]
            if self.args.shared_cfg.rot_6d:
                rot = euler_to_rot_6d(rot)
            action = np.concatenate([action[:, :3], rot, action[:, -1:]], axis=-1)
            if self.args.shared_cfg.use_delta_action:
                action = convert_delta_action(action[None], proprio.numpy())
                action = torch.tensor(action).float()
            else:
                action = torch.tensor(action)[None].float()

        # NOTE: here we didn't implement 1) support for eos prediction 2) support for euler outputs 
        obs = {
            "observation": image_vec,
            "proprio": proprio,
            "action": action
        }
        return obs

    def __call__(
        self, 
        side_image: Union[PIL.Image.Image, List[PIL.Image.Image]], 
        wrist_image : Union[PIL.Image.Image, List[PIL.Image.Image]], 
        proprio : Union[np.ndarray], 
        action : Optional[np.ndarray] = None,
        abs_gripper_control=False, # ignore temporal essembling for gripper control
        binary_gripper=False, # discretize the gripper control 
        use_temporal=True, # temporal essembling 
        teacher_forcing=False, # use ground truth action for prediction
    ):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            side_image (PIL.Image.Image or List[PIL.Image.Image]): side camera image
            wrist_image (PIL.Image.Image or List[PIL.Image.Image]): wrist camera image
            proprio (np.ndarray): proprioceptive information
            action (np.ndarray): action information
        """ 
        obs = self.prepare_observations(side_image, wrist_image, proprio, action)
        for k, v in obs.items():
            if v is not None:
                obs[k] = v.to(self.device, non_blocking=True)
        action = self.model.get_action_eval(
            obs,
            abs_gripper_control=abs_gripper_control,
            binary_gripper=binary_gripper,
            use_temporal=use_temporal,
            teacher_forcing=teacher_forcing,
        )
        if self.args.shared_cfg.rot_6d:
            rotation = torch.tensor(rot_6d_to_euler(action[3:-1].cpu().numpy())).to(action.device).squeeze()
            action = torch.cat([action[:3], rotation, action[-1:]])
        action = action.cpu().numpy()
        return action