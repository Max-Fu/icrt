import os
import yaml 
import json
import tyro 
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

import timm

from icrt.data.dataset import SequenceDataset
import icrt.util.misc as misc
from icrt.util.args import ExperimentConfig
from icrt.util.model_constructor import model_constructor
from icrt.data.utils import rot_6d_to_euler

def main(
    train_yaml_path : Union[str, Path],
    checkpoint_path : Union[str, Path],
    demo_index : int = 0, 
    train_val : str = "val",
    use_gt_action : bool = True, 
    vis_pred_step : int = 0,
):
    """Plotting the predicted actions vs the ground truth actions

    Args: 
        train_yaml_path: str, path to the yaml file containing the training configuration
        checkpoint_path: str, path to the checkpoint to load
        demo_index: int, index of the demo to plot
        train_val: str, "train" or "val", whether to use the train or val set
        use_gt_action: bool, whether to use the ground truth action or predicted action for the next step
        vis_pred_step: int, the step to visualize the predicted action
    """
    # we evaluate using the following pretrained_path
    args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader) 

    assert 0 <= vis_pred_step < args.shared_cfg.num_pred_steps, f"vis_pred_step {vis_pred_step} must be between 0 and num_pred_steps {args.shared_cfg.num_pred_steps}"

    model_output_dir = args.logging_cfg.output_dir
    # creating the output directory and logging directory for test
    args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, "test_output")
    args.logging_cfg.log_dir = args.logging_cfg.output_dir
    output_dir = args.logging_cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Logging results to ", output_dir)
    
    # dump the args into a yaml file 
    with open(os.path.join(output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

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
    demo_sequences = dataset[demo_index]

    # get the unconverted euler actions as well 
    dataset.rot_6d = False 
    demo_sequences_euler = dataset[demo_index]
    
    # reset the rotation back
    dataset.rot_6d = args.shared_cfg.rot_6d 

    # add batch dimension and move them to device
    for key in demo_sequences:
        if demo_sequences[key] is not None:
            demo_sequences[key] = demo_sequences[key][None]
    length = args.shared_cfg.seq_length
    
    predictions = []

    # using kv cache 
    first_obs = True
    start_pos = 0
    # for i in tqdm(range(2, length + 1)):
    for i in tqdm(range(1, length + 1)): 
        new_seq = {key : demo_sequences[key][:, max(i-2, 0):i].to(device).contiguous().float() for key in demo_sequences} 
        if first_obs:
            new_seq['action'] = None
        else:
            if use_gt_action:
                new_seq['action'] = new_seq["action"][:, :1] # use gt action
            else:
                new_seq['action'] = torch.tensor(full_action[None]).to(device).float() # use predicted action
            # process action, proprio (only take in one step observation)
            new_seq['action'] = new_seq['action'][:, :, 0] # we feed in the current action (B, 1, action_dim)
        new_seq['proprio'] = new_seq['proprio'][:, :, 0] # and we feed in the current proprio (B, 1, proprio_dim)

        with torch.no_grad():
            predicted, eos = model.forward_inference(new_seq, start_pos) # (B, num_pred_steps, action_dim), (B, num_pred_steps)
            eos = torch.zeros((predicted.shape[0], predicted.shape[1])) # we don't care about eos here

        predicted, eos = predicted[:, vis_pred_step], eos[:, vis_pred_step] # (B, action_dim), (B)
        predicted, eos = predicted.cpu().numpy().flatten(), eos.cpu().numpy().flatten()
        full_action = np.concatenate([predicted[None], eos[None]], axis=-1) # 1, action_dim
        predictions.append(full_action)

        if first_obs:
            first_obs = False 
        else:
            start_pos += 2
        
        # pred_action, pred_eos = predicted
    predictions = np.concatenate(predictions, axis=0)
    # we visualize the first action that needs to be predicted
    gt_act = demo_sequences['action'][0, :, 0].cpu().numpy()

    action_dim = gt_act.shape[-1]
    for j in range(action_dim):
        plt.plot(gt_act[:,j], label=f"action_{j}")
        if j == action_dim-1:
            values = 1/(1 + np.exp(-predictions[:,j]))
            plt.plot(values, label=f"predicted_action_{j}")
        else:
            plt.plot(predictions[:,j], label=f"predicted_action_{j}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"action_predicted_vs_gt_{j}.png"))
        plt.close()

    if args.shared_cfg.rot_6d :
        # convert the rotation into euler angles and plot again 
        predictions_rot = predictions[:, 3:9] 
        gt_rot = gt_act[:, 3:9]

        predictions_rot_euler = rot_6d_to_euler(predictions_rot)
        gt_rot_euler = rot_6d_to_euler(gt_rot)

        original_euler = demo_sequences_euler["action"][:, 3:6]

        for j in range(3): 
            plt.plot(gt_rot_euler[:,j], label=f"rot_{j}")
            plt.plot(predictions_rot_euler[:,j], label=f"predicted_rot_{j}")
            plt.plot(original_euler[:,j], label=f"original_rot_{j}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"rot_predicted_vs_gt_{j}.png"))
            plt.close()

if __name__ == '__main__':
    tyro.extras.set_accent_color("yellow")
    tyro.cli(main)
