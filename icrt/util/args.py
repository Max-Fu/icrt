import dataclasses
from typing import Literal, Optional, Tuple, Union
import enum
import pathlib

import tyro

@dataclasses.dataclass
class DatasetConfig: 
    # Dataset config path 
    dataset_json : str

    # Order the data episodes by language descriptions
    sort_by_lang : bool = True

    # Enforce only one task is being seen in a batch
    task_barrier : bool = True 

    # each step is only seen once in each episode 
    skip_step : bool = False

    # variance of the noise added to proprioception
    proprio_noise : float = 0.005

    # variance of the noise added to action 
    action_noise : float = 0.0

    # training on one or more specific tasks
    task_names : Optional[Tuple[str, ...]] = None

    # add vision data aug 
    vision_aug : bool = True
    
    # rebalance tasks 
    rebalance_tasks : bool = True

    # each batch contains data that is non overlapping (i.e. for each epoch the same state action does not appear twice)
    non_overlapping : Union[bool, int] = False

    # enable repeats of trajectory 
    num_repeat_traj : int = 1

    # shuffle repeat trajectory 
    shuffle_repeat_traj : bool = True
        
    #number of steps to weight
    num_weighted_steps : float = 30

    # use goal condition 
    goal_conditioned : bool = False

    # use a fraction of the dataset by task (0.0 to 1.0)
    dataset_fraction : float = 1.0

@dataclasses.dataclass
class VisionEncoderConfig: 
    # vision encoder type (or path to checkpoint)
    vision_encoder : str = "vit_base_patch16_224.mae"

    # Whether to use a randomly initialized vision encoder instead of pretrained weights
    vision_nonpretrained : bool = False 

    # Whether to unfreeze the vision encoder
    vision_unfreeze_all : bool = False 

    # Whether to use LoRA in the vision encoder 
    vision_lora : bool = False

    # Rank of LoRA layers
    vision_lora_rank : int = 8 

    # Number of blocks unfrozen in the vision encoder 
    vision_unfreeze_last_n : int = 0 

@dataclasses.dataclass
class PolicyConfig: 
    # path to LLaMA pretrained checkpoint
    llama_ckpt_dir : str = "/home/mfu/checkpoints/llama-2/llama-2-7b"

    # uses different linear layers for attention pooling 
    multikv_attn_pool : bool = False

    # Number of heads for adapter
    adapter_num_heads : int = 8 

    # Adapter MLP ratio 
    adapter_mlp_ratio : float = 4.0 

    # Weight for action loss 
    loss_w_action : float = 1.0 
    
    # add camera positional embeddings
    camera_pos_emb : bool = False 

    # add modality positional embeddings
    modality_pos_emb : bool = False

    # Separate encoder adapter for different cameras
    separate_camera_adapter : bool = True 

    # Rank of LoRA layers for Llama 
    lora_rank : int = 4 

    # Layer indices to apply LoRA
    lora_layer_idxs : Optional[Tuple[int, ...]] = None 

    # Phase of training 
    phase : Literal["pretrain", "finetune"] = "pretrain"

    # path to checkpoint from pretrain stage
    pretrained_path : Optional[str] = None

    # predict only action 
    pred_action_only : bool = True

    # remove proprio from input 
    remove_proprio : bool = False

    # no loss on prompt
    no_prompt_loss : bool = False

    # Prediction head, can be one of "mlp", "gmm", "diffusion"
    decoder_pred_head : Literal["mlp", "gmm", "diffusion"] = "mlp"

    # use kl div loss
    kl_div_loss : bool = False

    # loss scaler 
    scale_loss : float = 1.0
    
    # load llama
    load_llama : bool = True
    
    # step weight
    step_weight : float = 1.0

    # train llama from scratch
    scratch_llama_config : Optional[str] = None

    # training diffusion steps:
    num_train_diffusion_steps : Optional[int] = None

    # inference diffusion steps
    num_inference_diffusion_steps : Optional[int] = None 

@dataclasses.dataclass
class ModelConfig: 
    # Policy (llama + adapter) configuration
    policy_cfg : PolicyConfig

    # vision encoder config 
    vision_encoder_cfg : VisionEncoderConfig

@dataclasses.dataclass
class OptimizerConfig: 
    # weight decay (default: 0.01) 
    weight_decay : float = 0.01

    # learning rate (absolute lr)
    lr : Optional[float] = 5e-4

    # base learning rate: absolute_lr = base_lr * total_batch_size / 256 
    blr : float = 1e-3 

    # lower lr bound for cyclic schedulers that hit 0
    min_lr : float = 0.0 

    # epochs to warmup LR
    warmup_epochs : float = 40

@dataclasses.dataclass
class TrainerConfig:
    # number of epochs 
    epochs : int = 400

    # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    accum_iter : int = 8

    # pin memory for dataloader
    pin_memory : bool = True

    # number of workers for dataloader 
    num_workers : int = 20 


@dataclasses.dataclass
class SharedConfig:
    # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
    batch_size : int = 1

    # Use 6DoF Rotation 
    rot_6d : bool = True 

    # number of frames in a sequence 
    seq_length : int = 512

    # seed for random number generators
    seed : int = 0
    
    # start epoch 
    start_epoch : int = 0

    # frequency of saving checkpoint 
    save_every : int = 5

    # resume from checkpoint 
    resume : Optional[str] = None 

    # Number of stages for progressive training 
    num_stages : int = 1

    # split epoch into k different epochs 
    split_epoch : int = 1 

    # Number of cameras 
    num_cameras : int = 2 

    # Number of predicted action steps 
    num_pred_steps : int = 16
    
    # use delta action
    use_delta_action : bool = True

    # scale action with calculated action statistics (json file)
    scale_action : Optional[str] = None
    
@dataclasses.dataclass
class LoggingConfig:
    # path where to save, empty for no saving
    output_dir: str = "./output"

    # path where to save tensorboard logs 
    log_dir : Optional[str] = None 

    # log name (for wandb)
    log_name : Optional[str] = None

@dataclasses.dataclass
class ExperimentConfig: 
    # Dataset configuration
    dataset_cfg: DatasetConfig

    # Model configuration
    model_cfg: ModelConfig

    # Optimizer configuration
    optimizer_cfg: OptimizerConfig

    # Shared configuration
    shared_cfg: SharedConfig

    # Logging configuration 
    logging_cfg: LoggingConfig

    # trainer configuration
    trainer_cfg: TrainerConfig

    # train or eval 
    train : bool = True

    # number of distributed processes (required by torch distributed)
    world_size: int = 1

    # local rank of the process (required by torch distributed)
    local_rank: int = -1

    # distributed training on the optimizer (required by torch distributed)
    dist_on_itp: bool = False

    # distributed training url (required by torch distributed)
    dist_url: str = 'env://'

    # device to use for training / testing (required by torch distributed)
    device : str = "cuda"

    # load config. instead of using command line arguments, load from a config file
    load_config: Optional[str] = None

if __name__ == "__main__": 
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)