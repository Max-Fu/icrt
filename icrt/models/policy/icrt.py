# test script: 
# CUDA_VISIBLE_DEVICES=4 python -m llama.icrt

import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm 
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from diffusers import DDIMScheduler, DDPMScheduler

from typing import Union, Literal, Optional
from .llama import Transformer, ModelArgs
from .action_head import MLPHead, GMMHead, DiffusionHead
from icrt.models.backbones.encoders import VisionEncoder, VisionEncoderCNN, AttentionPool, MultiKVAttentionPool
from icrt.models.losses import losses
from icrt.data.utils import convert_abs_action, convert_delta_action, scale_action, unscale_action, load_json

class ICRT(nn.Module):
    """ 
    In context robot learning with transformer intialized by llama
    """
    # parameters that are not used in upstream code
    attn_latent_len : int = 1
    sa_loss_fn = "l1" # state action loss
    # eos_loss_fn = "bce_with_logits"
    loss_w_proprio : float = 1.
    # loss_w_eos : float = 1.
    decoder_hidden_features : int = 128
    # adding gmm hyperparameters
    num_gmm_components = 3
    # steps to give additional step weight

    # adding diffusion hyperparameters
    num_inference_diffusion_steps = 100 
    num_train_diffusion_steps : int = 100
    diffusion_beta_start : float = 0.0001 
    diffusion_beta_end : float = 0.02
    diffusion_beta_schedule : str = "squaredcos_cap_v2"
    
    def __init__(
        self, 
        llama_ckpt_dir : str, # path to llama checkpoint
        vision_encoder : Union[VisionEncoder, VisionEncoderCNN], # vision encoder
        phase : str = "pretrain", # phase of training
        num_cameras : int = 2, 
        proprio_dim : int = 8, # cartesian + gripper (7 if euler or axis angle, 8 if quarternion)
        action_dim : int = 8, # cartesian + gripper + eos
        adapter_mlp_ratio : int = 4,
        adapter_num_heads : int = 8,
        multikv_attn_pool : bool = False,
        loss_w_action : float = 1., 
        lora_rank : int = 4,
        camera_pos_emb : bool = False,
        modality_pos_emb : bool = False,
        separate_camera_adapter : bool = False,  
        seq_length : int = 256, # number of (s,a) pairs
        rot_6d : bool = False,
        train : bool = True, 
        max_batch_size : int = 2, 
        num_pred_steps : int = 1, 
        pred_action_only : bool = False,
        remove_proprio : bool = False,
        no_prompt_loss: bool = False,
        decoder_pred_head : Literal["mlp", "gmm", "diffusion"] = "mlp",
        use_delta_action: bool = False,
        kl_div_loss: bool = False,
        scale_loss : float = 1.0, 
        load_llama: bool = True,
        step_weight: float = 1.0, #weight to give to the first 10 steps
        scratch_llama_config : Optional[str] = None, # config for training llama from scratch, 
        num_train_diffusion_steps : Optional[int] = None,
        num_inference_diffusion_steps : Optional[int] = None,
        scale_action : Optional[str] = None,
    ):
        super().__init__()

        # define language model parameters
        self.scratch_llama_config = scratch_llama_config
        if self.scratch_llama_config is not None: 
            llama_config_path = self.scratch_llama_config
        else: 
            llama_config_path = os.path.join(llama_ckpt_dir, "params.json")
        with open(llama_config_path, "r") as f:
            params = json.loads(f.read())
        bias_lora = phase == "finetune"
        # args.seq_length is the number (s,a) pairs, so needs to multiply by 2
        max_batch_size = max_batch_size
        model_args = ModelArgs(
            max_seq_len=seq_length*2, 
            max_batch_size=max_batch_size, 
            w_bias=bias_lora, 
            w_lora=bias_lora, 
            lora_rank=lora_rank, 
            **params
        ) 
        self.latent_dim = model_args.dim
        print(f"language model args: {model_args}")

        # set up diffusion parameters if they are defined 
        if num_train_diffusion_steps is not None:
            self.num_train_diffusion_steps = num_train_diffusion_steps
        if num_inference_diffusion_steps is not None:
            self.num_inference_diffusion_steps = num_inference_diffusion_steps

        # vision encoder 
        self.vision_encoder = vision_encoder
        self.vision_out_dim = self.vision_encoder.out_dim()
        self.vision_finetune = vision_encoder.finetune 
        
        # randomly initialize camera positional embeddings
        self.num_cameras = num_cameras
        self.camera_pos_emb = camera_pos_emb
        if camera_pos_emb:
            self.icrt_camera_pos_emb = nn.Parameter(torch.randn(num_cameras, 1, self.vision_out_dim))
            nn.init.trunc_normal_(self.icrt_camera_pos_emb, std=0.2)

        # proprioception encoder
        self.proprio_dim = proprio_dim

        self.remove_proprio = remove_proprio
        if not self.remove_proprio:
            self.icrt_proprio_encoder = Mlp(in_features=self.proprio_dim, out_features=self.vision_out_dim)

        # attention pooling 
        assert self.attn_latent_len == 1, "currently causal attention only supports latent_len=1"
        attn_pool_cls = MultiKVAttentionPool if multikv_attn_pool else AttentionPool

        self.separate_camera_adapter = separate_camera_adapter and num_cameras > 1
        if self.separate_camera_adapter: 
            self.icrt_attn_pooling = nn.ModuleList(
                [attn_pool_cls(
                    self.vision_out_dim, out_features=self.latent_dim // num_cameras, mlp_ratio=adapter_mlp_ratio,
                    latent_len=self.attn_latent_len, num_heads=adapter_num_heads,
                ) for _ in range(num_cameras)]
            )
            self.padding = self.latent_dim - self.latent_dim // num_cameras * num_cameras
        else:
            self.icrt_attn_pooling = attn_pool_cls(
                self.vision_out_dim, out_features=self.latent_dim, mlp_ratio=adapter_mlp_ratio,
                latent_len=self.attn_latent_len, num_heads=adapter_num_heads,
            )
        self.icrt_vision_norm = nn.LayerNorm(normalized_shape=self.vision_out_dim, eps=1e-6) # consistent with timm ViT Formulation
        
        # flag for prediction action only: if this is true, no proprio prediction is needed
        self.pred_action_only = pred_action_only 
        self.no_prompt_loss = no_prompt_loss
        
        self.step_weight = step_weight

        # decoding to proprio space
        self.num_pred_steps = num_pred_steps
        if not self.pred_action_only:
            self.action_dim = action_dim
        else: 
            # we do not predict eos anymore 
            self.action_dim = action_dim - 1
        self.icrt_action_encoder = Mlp(in_features=self.action_dim, out_features=self.latent_dim)

        self.rot_6d = rot_6d
        self.use_delta_action = use_delta_action

        # aggregate state and action
        self.latent_len = self.attn_latent_len + 1

        # positional embedding for combination of (f_v and f_p) and action (optional, defined by flag)
        self.modality_pos_emb = modality_pos_emb
        if self.modality_pos_emb:
            print("training modality positional embedding")
            self.icrt_pos_emb_state = nn.Parameter(torch.randn(self.latent_dim))
            self.icrt_pos_emb_act = nn.Parameter(torch.randn(self.latent_dim))

            # initialize them 
            nn.init.trunc_normal_(self.icrt_pos_emb_state, std=0.2)
            nn.init.trunc_normal_(self.icrt_pos_emb_act, std=0.2)

        # constructing decoder for proprio and action 
        self.decoder_pred_head = decoder_pred_head
        self._construct_decoder_heads()

        # transformer 
        print("initializing main transformer ...")
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        print("initialization completed. ")

        # we enforce loading to llama checkpoint
        if self.scratch_llama_config is None and load_llama: # this is deprecated
            ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
            for ckpt in tqdm(ckpts, desc="Loading LLaMA ckpt"):
                ckpt = torch.load(ckpt, map_location='cpu')
                names = self.llama.state_dict().keys()
                ckpt_names = ckpt.keys()
                for n in ckpt_names:
                    if n not in names:
                        print(f"Warning: {n} not in llama model")
                self.llama.load_state_dict(ckpt, strict=False)

        # # 6. training criterion
        # self.state_action_loss = losses[self.sa_loss_fn](reduction="none")
        # self.eos_loss = losses[self.eos_loss_fn](reduction="none")
        self.kl_div_loss = kl_div_loss
        self.kl_loss = losses['kl_div']
        if self.kl_div_loss:
            self.loss_w_kl = 1e-5
        else:
            self.loss_w_kl = 0.0

        # 7. weighs 
        if self.pred_action_only:
            # we assign no weight to proprio and eos prediction
            self.loss_w_proprio = 0
        
        self.scale_loss = scale_loss
        total_weight = sum([loss_w_action, self.loss_w_proprio, self.loss_w_kl])
        self.loss_w_action = loss_w_action / total_weight
        self.loss_w_proprio = self.loss_w_proprio / total_weight
        self.loss_w_kl = self.loss_w_kl / total_weight
        
        self.phase = phase

        # whether or not to rescale action 
        self.scale_action = scale_action
        if self.scale_action is not None: 
            print("Loaded action scaling from ", self.scale_action)
            self.scale_action = load_json(self.scale_action)
            self.scale_action = {k: torch.tensor(v) for k, v in self.scale_action.items()}
        else:
            print("No action scaling applied")
            self.scale_action = None

        if train:
            self.set_default_trainability(self.phase)
        else:
            self.eval()

    def _construct_decoder_heads(self):
        """
        construct the decoder heads for action, proprio, and eos prediction
        """
        def _action_head_constructor(
            head_type : Literal["mlp", "gmm", "diffusion"],
            input_dim : int, 
            hidden_features : int,
            output_dim : int, 
            loss_fn : Optional[nn.Module] = None,
        ):
            """
            construct the prediction head
            Args:
                head_type: str, type of the prediction head, one of "mlp", "gmm", "diffusion"
                input_dim: int, input dimension
                hidden_features: int, hidden dimension
                output_dim: int, output dimension
                loss_fn: nn.Module, loss function
            """
            if head_type == "mlp":
                return MLPHead(
                    input_dim=input_dim,
                    hidden_features=hidden_features,
                    output_dim=output_dim,
                    loss_fn=loss_fn,
                )
            elif head_type == "gmm":
                print("ignoring loss_fn for GMM head")
                return GMMHead(
                    input_dim=input_dim, 
                    hidden_features=hidden_features,
                    output_dim=output_dim,
                    num_components=self.num_gmm_components,
                )
            elif head_type == "diffusion":
                print("ignoring loss_fn for diffusion head")
                inference_noise_scheduler = DDPMScheduler(
                    self.num_train_diffusion_steps, 
                    self.diffusion_beta_start, 
                    self.diffusion_beta_end, 
                    self.diffusion_beta_schedule,
                    clip_sample = True
                )
                train_noise_scheduler = DDPMScheduler(
                    self.num_inference_diffusion_steps, 
                    self.diffusion_beta_start, 
                    self.diffusion_beta_end, 
                    self.diffusion_beta_schedule,
                    clip_sample = True
                )
                return DiffusionHead(
                    input_dim=input_dim, 
                    hidden_features=hidden_features,
                    output_dim=output_dim,
                    train_noise_scheduler = train_noise_scheduler,
                    inference_noise_scheduler = inference_noise_scheduler,
                    inference_steps=self.num_inference_diffusion_steps,
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
        
        if not self.pred_action_only:
            self.icrt_proprio_decoder = _action_head_constructor(
                head_type=self.decoder_pred_head,
                input_dim=self.latent_dim,
                hidden_features=self.decoder_hidden_features,
                output_dim=self.proprio_dim * self.num_pred_steps,
                loss_fn=losses[self.sa_loss_fn](reduction="none"),
            )

        self.icrt_action_decoder = _action_head_constructor(
            head_type=self.decoder_pred_head,
            input_dim=self.latent_dim,
            hidden_features=self.decoder_hidden_features,
            output_dim=self.action_dim * self.num_pred_steps,
            loss_fn=losses[self.sa_loss_fn](reduction="none"),
        )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(ICRT, self).state_dict(destination, prefix, keep_vars) 
        trainable_params = self.get_trainable_params(self.phase) # trainable_params points to different tensor as state_dict
        new_state_dict = OrderedDict()
        for k in trainable_params:
            new_state_dict[k] = state_dict[k]
        return new_state_dict

    def get_trainable_params(self, phase='finetune'):
        trainable = {}
        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                elif self.vision_finetune and name.startswith("vision_encoder.") and para.requires_grad:
                    trainable[name] = para
                elif name.startswith("icrt_"):
                    trainable[name] = para 
        elif phase == 'pretrain':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if self.scratch_llama_config is not None:
                        trainable[name] = para
                    elif 'gate' in name:
                        trainable[name] = para
                elif name.startswith("icrt_"):
                    trainable[name] = para 
                elif self.vision_finetune and name.startswith("vision_encoder.") and para.requires_grad:
                    trainable[name] = para
        elif phase == 'frozen':
            pass
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        return trainable

    def set_default_trainability(self, phase='finetune'):
        for key, value in self.named_parameters():
            # vision encoder trainability is set upon construction
            if key.startswith("vision_encoder."):
                continue
            value.requires_grad = False
        for key, value in self.get_trainable_params(phase).items():
            value.data = value.data.float()
            value.requires_grad = True

    def preprocessing(
        self, 
        observation : torch.Tensor, 
        proprio : torch.Tensor,
        action : torch.Tensor,
    ) -> torch.Tensor:
        """
        This function preprocesses the observation, proprioception, and action data for the ICRT model.
        
        Parameters:
        observation : B, T, N, 3, H, W (batch size, timesteps, num_cameras, 3, height, width)
        proprio : B, T, proprio_dim (batch size, timesteps, proprio_dim)
        action : B, T, action_dim (batch size, timesteps, action_dim)
        robot / dataset type (optional) : B, T, 1 (batch size, timesteps, 1)
        
        Returns:
        torch.Tensor: The preprocessed data, a tensor of shape (B, T * (latent_len+1), self.latent_dim).
        """
        # proprio processing
        if not self.remove_proprio:
            f_prop = self.icrt_proprio_encoder(proprio) # B, T, self.vision_out_dim
        
        # observation processing
        f_obs = self.vision_encoder(observation) # B, T, N, K, self.vision_out_dim (k is number of patches + cls token)
        f_obs = self.icrt_vision_norm(f_obs)
        if self.camera_pos_emb:
            f_obs = f_obs + self.icrt_camera_pos_emb # add camera positional embeddings
        
        if self.separate_camera_adapter:
            if self.remove_proprio:
                f_obs = [self.icrt_attn_pooling[i].forward_visual(f_obs[:, :, i]) for i in range(self.num_cameras)]
            else:
                f_obs = [self.icrt_attn_pooling[i].combine_forward(f_obs[:, :, i], f_prop) for i in range(self.num_cameras)]
            f_s = torch.concat(f_obs, dim=-1)
            # pad at the last dimension
            if self.padding > 0:
                f_s = F.pad(f_s, (0, self.padding), "constant", 0)

        else:
            f_obs = f_obs.view(f_obs.shape[0], f_obs.shape[1], -1, f_obs.shape[-1]) # B, T, N*K, self.vision_out_dim
            # state extraction 
            if self.remove_proprio:
                f_s = self.icrt_attn_pooling.forward_visual(f_obs) # B, T, latent_len, self.latent_dim
            else:
                f_s = self.icrt_attn_pooling.combine_forward(f_obs, f_prop) # B, T, latent_len, self.latent_dim

        # action processing
        f_a = self.icrt_action_encoder(action) # B, T, D
        f_a = f_a[:, :, None, :] # B, T, 1, self.latent_dim

        # interweave state and action on the temporal dimension
        if self.modality_pos_emb:
            f_s = f_s + self.icrt_pos_emb_state
            f_a = f_a + self.icrt_pos_emb_act

        f_sa = torch.cat([f_s, f_a], dim=2) # B, T, latent_len+1, self.latent_dim
        f_sa = f_sa.view(f_sa.shape[0], -1, f_sa.shape[-1]) # B, T * (latent_len+1), self.latent_dim
        return f_sa

    def postprocessing(self, f_sa : torch.Tensor, B : int, T : int) -> tuple:
        """
        Predicting action, proprio, and eos from the output of the transformer.
        f_sa: output of the transformer, shape: [B, T, latent_len+1, self.latent_dim]
        
        Returns:
        tuple: (out_action, out_proprio, out_eos)
        out_actions: shape [B, T, num_pred_steps, action_dim]
        out_proprio: shape [B, T, num_pred_steps, proprio_dim]
        out_eos: shape [B, T, num_pred_steps]
        """
        # parse output into state and action 
        out_a = f_sa[:, :, :-1, :]
        out_s = f_sa[:, :-1, -1, :] # we ignore the last state prediction, as there is no action to predict

        # decoding to actions 
        out_a = out_a.view(B*T, -1, self.latent_dim)
        out_a = self.icrt_action_decoder(out_a).squeeze().view(B, T, self.action_dim * self.num_pred_steps).view(B, T, self.num_pred_steps, self.action_dim)
        
        # decoding to proprio
        if not self.pred_action_only:
            out_proprio = self.icrt_proprio_decoder(out_s).view(B, T - 1, self.num_pred_steps, self.proprio_dim)
            out_eos = out_a[:, :, :, -1]
            out_action = out_a[:, :, :, :-1]
        else:
            out_proprio = None
            out_eos = None
            out_action = out_a
        return out_action, out_proprio, out_eos
    
    def preprocessing_attention(self, 
        observation : torch.Tensor, 
        proprio : torch.Tensor,
        action : torch.Tensor,) -> torch.Tensor:
        f_prop = self.icrt_proprio_encoder(proprio) # B, T, self.vision_out_dim
        
        # observation processing
        f_obs = self.vision_encoder(observation) # B, T, N, K, self.vision_out_dim (k is number of patches + cls token)
        f_obs = self.icrt_vision_norm(f_obs)
        if self.camera_pos_emb:
            f_obs = f_obs + self.icrt_camera_pos_emb # add camera positional embeddings
        
        f_obs = f_obs.view(f_obs.shape[0], f_obs.shape[1], -1, f_obs.shape[-1]) # B, T, N*K, self.vision_out_dim
        f_s = self.icrt_attn_pooling.forward_attention(f_obs, f_prop) # B, T, latent_len, self.latent_dim
        
        return f_s

    def forward_vision_attention(self, sequences : torch.Tensor) -> torch.Tensor:
        observation = sequences['observation']
        proprio = sequences['proprio']
        action = sequences['action']

        if self.pred_action_only:
            # remove eos 
            action = action[..., :self.action_dim]
        
        # get dimensions 
        B, T = observation.shape[:2]
    
        # the inputs are only single step 
        proprio_in = proprio[:, :, 0] # B, T, num_pred_steps, proprio_dim -> B, T, proprio_dim
        action_in = action[:, :, 0] # B, T, num_pred_steps, proprio_dim -> B, T, proprio_dim

        # preprocessing observations, proprio and actions
        return self.preprocessing_attention(observation, proprio_in, action_in)

    def forward(self, sequences : dict) -> torch.Tensor:
        """
        sequences: dict of tensors
            observation : B, T, N, 3, H, W (batch size, timesteps, num_cameras, 3, height, width)
            proprio : B, T, num_pred_steps, proprio_dim (batch size, timesteps, proprio_dim)
            action : B, T, num_pred_steps, action_dim (batch size, timesteps, action_dim)
        return loss: torch.tensor 
        """
        observation = sequences['observation']
        proprio = sequences['proprio']
        action = sequences['action']
        prompt_mask = sequences['prompt_mask']
        weight_mask = sequences['weight_mask'] # B, 1

        if self.pred_action_only:
            # remove eos 
            action = action[..., :self.action_dim]
        
        # get dimensions 
        B, T = observation.shape[:2]
    
        # the inputs are only single step 
        proprio_in = proprio[:, :, 0] # B, T, num_pred_steps, proprio_dim -> B, T, proprio_dim
        action_in = action[:, :, 0] # B, T, num_pred_steps, proprio_dim -> B, T, proprio_dim

        # preprocessing observations, proprio and actions
        f_sa = self.preprocessing(observation, proprio_in, action_in)

        # if self.no_prompt_loss:
        #     B, seq_length = f_sa.shape[0], f_sa.shape[1]
        #     mask = torch.full((B, 1, seq_length, seq_length), float(1.0), device=f_sa.device)
        #     mask = torch.triu(mask, diagonal=1).type_as(f_sa)
        #     reshaped_prompt_mask = prompt_mask.repeat_interleave(2,dim=1)[:,None,None,:].repeat(1,1,seq_length,1)
        #     mask = mask * reshaped_prompt_mask
        #     mask = mask.masked_fill(mask == 1, float('-inf'))
        # else:
        #     mask = None
        mask = None

        # forward LM 
        out = self.llama(f_sa,mask) # B, T * (latent_len+1), self.latent_dim

        # parse output into state and action 
        out = out.view(B, T, -1, self.latent_dim) # B, T, latent_len+1, self.latent_dim

        # predicting action, proprio, and eos
        out_a = out[:, :, :-1, :].view(B*T, (self.latent_len - 1) * self.latent_dim)
        out_s = out[:, :-1, -1, :].reshape(B*(T-1), self.latent_dim)

        if self.kl_div_loss:
            #torch select accoriding to prompt mask
            merged_out = out.permute(0,1,3,2).reshape(B,T,-1) # s,a is interleaved in this way

            prompt_out = merged_out*(1-prompt_mask)[...,None]
            prompt_mean = torch.sum(prompt_out, dim=1)/torch.sum(1-prompt_mask, dim=1)
            prompt_std = torch.square(prompt_out - prompt_mean[:,None])*(1-prompt_mask[...,None])
            # prompt_std = torch.sqrt(torch.sum(prompt_std, dim=1))/torch.sum(1-prompt_mask, dim=1)
            prompt_std = torch.ones_like(prompt_mean)
            prompt_dist = torch.distributions.Normal(prompt_mean, prompt_std+1e-5)

            pred_out = merged_out*prompt_mask[...,None]
            pred_mean = torch.sum(pred_out, dim=1)/torch.sum(prompt_mask, dim=1)
            pred_std = torch.square(pred_out - pred_mean[:,None])*prompt_mask[...,None]
            # pred_std = torch.sqrt(torch.sum(pred_std, dim=1))/torch.sum(prompt_mask, dim=1)
            pred_std = torch.ones_like(pred_mean)
            pred_dist = torch.distributions.Normal(pred_mean, pred_std+1e-5)

            kl_loss = self.kl_loss(pred_dist, prompt_dist).mean()
        else:
            kl_loss = 0.0

        # calculate state loss # we can also some clever stop_grad 
        if not self.pred_action_only:
            # predict proprio as well
            proprio_target = proprio[:, 1:, :].reshape(B*(T-1), self.proprio_dim * self.num_pred_steps) 
            state_loss = self.icrt_proprio_decoder.loss(out_s, proprio_target)
        else:
            state_loss = torch.tensor(0.0, dtype=proprio.dtype,device=proprio.device)

        if self.scale_action is not None:
            action_target = scale_action(action, self.scale_action)

        action_target = action.reshape(B*T, self.action_dim * self.num_pred_steps)
        action_loss = self.icrt_action_decoder.loss(out_a, action_target)

        if self.no_prompt_loss:
            state_loss = state_loss * prompt_mask[:,:-1].reshape(B*(T-1),1)
            action_loss = action_loss * prompt_mask.view(B*T,1)

        if self.step_weight > 1:
            weights_for_steps = weight_mask*self.step_weight
            
            state_loss = state_loss * weights_for_steps[:,:-1].reshape(B*(T-1),1)
            action_loss = action_loss * weights_for_steps.view(B*T,1)
        
        if self.no_prompt_loss:
            state_loss = state_loss.sum()/(prompt_mask[:,:-1].sum() * self.proprio_dim * self.num_pred_steps + 1e-6)
            action_loss = action_loss.sum()/(prompt_mask.sum() * self.action_dim * self.num_pred_steps + 1e-6)
        else:
            state_loss = state_loss.mean()
            action_loss = action_loss.mean()

        # calculate total loss
        loss = self.loss_w_action * action_loss + self.loss_w_proprio * state_loss + self.loss_w_kl*kl_loss
        loss = self.scale_loss * loss
        loss_dict = {
            "action_loss": action_loss,
            "state_loss": state_loss,
            "kl_loss": kl_loss,
            "loss": loss, 
        }
        return loss, loss_dict

    def get_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_new_parameters(self):
        for name, param in self.named_parameters():
            if "icrt" in name:
                print("name: {}; num_params: {}".format(name, param.numel()))

    @torch.inference_mode()
    def forward_inference(self, sequences : dict, start_pos: int):
        """
        observations: B, T, N, 3, H, W (batch size, timesteps, num_cameras, 3, height, width)
        proprio: B, T, proprio_dim (batch size, timesteps, proprio_dim)
        action: B, T - 1, action_dim (batch size, timesteps, action_dim) # we don't have action for the last state
        (action: B, T, action_dim (batch size, timesteps, action_dim) # for state prediction)
        robot / dataset type (optional) : B, T, 1 (batch size, timesteps, 1)
        
        Returns:
        1) if observation temporal length = action temporal length + 1
            tuple: (out_action, out_eos)
            out_actions: shape [B, self.num_pred_steps, action_dim]
            out_eos: shape [B, self.num_pred_steps]

        2) if observation temporal length = action temporal length
            out_proprio: shape [B, self.num_pred_steps, proprio_dim]
        """
        observation = sequences['observation']
        proprio = sequences['proprio']
        action = sequences.get("action", None)
        prompt_mask = sequences.get("prompt_mask", None)

        if action is not None and self.pred_action_only:
            # need to remove eos 
            action = action[..., :self.action_dim]
        
        # get dimensions 
        B, T = observation.shape[:2]
        
        # padding actions so interweaving works 
        action_T = action.shape[1] if action is not None else 0
        assert T == action_T or T == action_T + 1, f"temporal length differs by {T - action_T}"
        if action_T == T - 1:
            if action is None: 
                action = torch.zeros(B, 1, self.action_dim, dtype=proprio.dtype).to(device=proprio.device)
            else:
                action = torch.concatenate([action, torch.zeros(B, 1, self.action_dim).to(device=action.device)], dim=1)

        # preprocessing observations, proprio and actions
        f_sa = self.preprocessing(observation, proprio, action)

        # if self.no_prompt_loss and prompt_mask is not None:
        #     B, seq_length = f_sa.shape[0], f_sa.shape[1]
        #     mask = torch.full((seq_length, seq_length), float(1.0), device=f_sa.device)
        #     mask = torch.triu(mask, diagonal=1).type_as(f_sa)
        #     reshaped_prompt_mask = prompt_mask[0].repeat_interleave(2)[None,:].repeat(seq_length,1)
        #     mask = mask * reshaped_prompt_mask
        #     mask = mask.masked_fill(mask == 1, float('-inf'))
        # else:
        #     mask = None
        mask = None


        # if self.no_prompt_loss and prompt_mask is not None:
        #     B, seq_length = f_sa.shape[0], f_sa.shape[1]
        #     mask = torch.full((seq_length, seq_length), float(1.0), device=f_sa.device)
        #     mask = torch.triu(mask, diagonal=1).type_as(f_sa)
        #     reshaped_prompt_mask = prompt_mask[0].repeat_interleave(2)[None,:].repeat(seq_length,1)
        #     mask = mask * reshaped_prompt_mask
        #     mask = mask.masked_fill(mask == 1, float('-inf'))
        # else:
        #     mask = None

        # forward LM
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = self.llama.forward_inference(f_sa, start_pos, mask) # B, T * (latent_len+1), self.latent_dim

        # parse the output of the model 
        out = out.view(B, T, -1, self.latent_dim) # B, T, latent_len+1, self.latent_dim


        if action_T == T - 1:
            # predicting action
            out_a_latent = out[:, -1, :-1, :].view(B, (self.latent_len - 1) * self.latent_dim) # B, latent_len, self.latent_dim
            out_a = self.icrt_action_decoder.pred(out_a_latent).view(B, self.num_pred_steps, self.action_dim)
            
            if not self.pred_action_only:
                # separate eos and action 
                out_eos = out_a[:, :, -1] # B, self.num_pred_steps
                out_action = out_a[:, :, :-1] # B, self.num_pred_steps, action_dim
            else:
                # there's no eos prediction 
                out_eos = None
                out_action = out_a

            if self.scale_action is not None:
                out_action_cpu = out_action.detach().cpu()
                out_action = unscale_action(out_action_cpu, self.scale_action).to(device=out_action.device)

            if self.rot_6d:
                a1 = out_action[:, :, 3:6]
                a2 = out_action[:, :, 6:9]
                b1 = F.normalize(a1, dim=-1)
                b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
                out_action = torch.cat([out_action[:, :, :3], b1, b2, out_action[:, :, -1:]], dim=-1)

            out_action = out_action.float()
            if out_eos is not None:
                out_eos = out_eos.float()

            return out_action, out_eos

        # else:
        #     assert not self.pred_action_only, "Prediction action only is not supported for proprio prediction"
        #     # predicting latent state
        #     out_s = out[:, -1, -1, :] # B, self.latent_dim
        #     out_proprio = self.icrt_proprio_decoder.pred(out_s).view(B, self.num_pred_steps, self.proprio_dim)
        #     return out_proprio.float()

    def reset(self, action_exec_horizon : int = 1) -> None:
        """Reset the model for inference.

        Args: 
            action_exec_horizon: int, the number of steps to execute the action for
        """
        assert action_exec_horizon <= self.num_pred_steps, f"Execution horizon must be no greater than prediction horizon"
        self.action_exec_horizon = action_exec_horizon
        self.action_queue = deque(maxlen=self.action_exec_horizon)
        self.start_pos = 0
        self.last_action = None
        self.last_observation = None
        self.first_obs = True

    @torch.inference_mode()
    def prompt(self, prompt_sequences : dict) -> None:
        """Prompt the model with a sequence of observations and actions.
        assume that the prompt is generated from the dataset 

        Args:
            prompt_sequences: dict, a dictionary containing the sequences of observations and actions.
                observation : B, T, N, 3, H, W (batch size, timesteps, num_cameras, 3, height, width)
                proprio : B, T, num_pred_steps, proprio_dim (batch size, timesteps, num_pred_steps, proprio_dim)
                action : B, T, num_pred_steps, action_dim (batch size, timesteps, num_pred_steps, action_dim)
        """
        # the model takes one step proprio and action as input
        if prompt_sequences["action"].shape == 4:
            prompt_sequences["action"] = prompt_sequences["action"][:, :, 0]
        if prompt_sequences["proprio"].shape == 4:
            prompt_sequences["proprio"] = prompt_sequences["proprio"][:, :, 0]

        prompt_len = prompt_sequences["action"].shape[1] + prompt_sequences["observation"].shape[1]
        batch_size, mask_len = prompt_sequences["action"].shape[0], prompt_sequences["action"].shape[1]
        # if self.no_prompt_loss:
        #     prompt_sequences["prompt_mask"] = torch.zeros(
        #         batch_size, mask_len, device=prompt_sequences["action"].device, dtype=prompt_sequences["action"].dtype
        #     )
        self.forward_inference(prompt_sequences, self.start_pos)
        self.start_pos += prompt_len
        self.first_obs = True
        
    @torch.inference_mode()
    def get_action(self, observation : dict) -> torch.Tensor:
        """Generating action from observations 

        Args:
            observation: dict, a dictionary containing the observations.
                "observation" : 1, 2, N, 3, H, W or 1, 1, N, 3, H, W (batch size, T=2, num_cameras, 3, height, width)
                "proprio" : 1, 2, proprio_dim or 1, 1, proprio_dim (batch size, T=2, proprio_dim)
            Note it does not contain action 
        
        Returns:
            torch.Tensor: the action generated by the model: action_dim, 
        """
        tmp = observation.copy()
        if self.last_observation is not None:
            for k in observation:
                observation[k] = torch.cat([self.last_observation[k], observation[k]], dim=1)
        self.last_observation = tmp
        observation["action"] = self.last_action

        action_sequence, eos_sequence = self.forward_inference(observation, self.start_pos)
        if len(self.action_queue) == 0:
            self.action_queue.extend(action_sequence[0, :self.action_exec_horizon])

        action = self.action_queue.popleft()
        self.last_action = torch.cat([action.clone().unsqueeze(0).unsqueeze(0), torch.zeros((1, 1, 1), device=action.device)], dim=-1) # add eos
        
        if self.first_obs:
            self.first_obs = False
        else:
            self.start_pos += 2
        
        return action
    
    @torch.inference_mode()
    def get_action_eval(self, observation, abs_gripper_control=False, use_temporal=True, binary_gripper=False, teacher_forcing=False):
        # if abs_gripper_control, then the current model prediction is used instead of the averaged gripper action
        tmp = observation.copy()
        if teacher_forcing:
            gt_action = observation["action"]
        if self.last_observation is not None:
            for k in observation:
                if self.last_observation[k] is not None:
                    observation[k] = torch.cat([self.last_observation[k], observation[k]], dim=1)
        self.last_observation = tmp

        if teacher_forcing:
            if observation["action"] is not None:
                observation["action"] = torch.cat([gt_action, torch.zeros((1, 1, 1), device=observation["action"].device)], dim=-1)
        else:
            observation["action"] = self.last_action

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            action_sequence, eos_sequence = self.forward_inference(observation, self.start_pos)
        action = action_sequence[0].squeeze()

        if binary_gripper:
            #!!!!!!!!!!!!!!!!!!!!binarize the gripper action
            print("raw_gripper_action", action[:,-1])
            action[:,-1] = action[:,-1] > 0.5
            
        #delta action
        if self.use_delta_action:
            delta_action = action.clone()
            action = convert_abs_action(delta_action[None].cpu().numpy(),observation['proprio'].cpu().numpy())
            action = torch.tensor(action,device=delta_action.device).squeeze().float()

        if abs_gripper_control:
            gripper_action = action[0, -1]

        if not use_temporal:
            if len(self.action_queue) == 0: 
                self.action_queue = deque(action[:self.action_exec_horizon])
            action = self.action_queue.popleft()
        else:
            new_actions = deque(action[:self.action_exec_horizon])
            self.action_queue.append(new_actions)
            actions_current_timestep = torch.empty((len(self.action_queue), action.size(1))).to(action.device)
            
            k = 0.05
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()
            exp_weights = torch.exp(k * torch.arange(actions_current_timestep.size(0))).to(action.device)
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions_current_timestep * exp_weights[:, None]).sum(dim=0)

        #convert to delta action
        if self.use_delta_action:
            last_action = convert_delta_action(action[None,None].cpu().numpy(),observation['proprio'].cpu().numpy())
            last_action = torch.tensor(last_action,device=action.device).squeeze().float()
        else:
            last_action = action.clone()

        self.last_action = torch.cat([last_action.clone().unsqueeze(0).unsqueeze(0), torch.zeros((1, 1, 1), device=last_action.device)], dim=-1)

        if self.first_obs:
            self.first_obs = False
        else:
            self.start_pos += 2
        
        if abs_gripper_control:
            action[-1] = gripper_action

        
        return action
