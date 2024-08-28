# Training ICRT 

After setting up the dataset, you can train ICRT. An example of the training script is described below. Fill in `/path/to/output_dir`, `log_name`, and `/path/to/vision/encoder.pth`. 
```bash 
python scripts/train.py --dataset-cfg.dataset-json /path/to/dataset_config.json --logging-cfg.output-dir /path/to/output_dir --logging-cfg.log-name log_name --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder /path/to/vision/encoder.pth --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 32 --trainer-cfg.accum-iter 32 --shared-cfg.batch-size 2
```
We support GPU parallelization with DDP: 
```bash
torchrun --nproc_per_node=8 --master_port=2450 scripts/train.py --dataset-cfg.dataset-json /path/to/dataset_config.json --logging-cfg.output-dir /path/to/output_dir --logging-cfg.log-name log_name --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder /path/to/vision/encoder.pth --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 32 --trainer-cfg.accum-iter 4 --shared-cfg.batch-size 2
```

To check what are the available flags, it is detailed in [args.py](icrt/util/args.py). We use [tyro](https://brentyi.github.io/tyro/) as the command-line interface, you can also try:
```bash 
python scripts/train.py -h
```

## Pre-training
For pre-training on DROID, we used the following config:
```bash
torchrun --nproc_per_node=8 --master_port=2450 scripts/train.py --dataset-cfg.dataset-json config/droid_10k_dataset_config.json --logging-cfg.output-dir /path/to/output_dir --logging-cfg.log-name pretrain_droid --optimizer-cfg.warmup-epochs 0.5 --trainer-cfg.epochs 4 --model-cfg.vision-encoder-cfg.vision-encoder /path/to/vision/encoder.pth --dataset-cfg.non-overlapping 50 --optimizer-cfg.lr 1e-3 --model-cfg.policy-cfg.phase pretrain --shared-cfg.save-every 1 --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json
```

## Finetuning 
We finetune on ICRT-MT. We take the last checkpoint of pre-training and finetune for 150 epochs. Usually we find that the model's performance is best around epoch 65-85. 
```bash
torchrun --nproc_per_node=8 --master_port=2450 scripts/train.py --dataset-cfg.dataset-json config/dataset_config.json --logging-cfg.output-dir /path/to/output_dir --logging-cfg.log-name finetune_icrt_mt --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder /path/to/vision/encoder.pth --dataset-cfg.num-repeat-traj 2 --model-cfg.policy-cfg.no-prompt-loss --model-cfg.policy-cfg.phase pretrain --model-cfg.policy-cfg.scratch-llama-config config/model_config/custom_transformer.json --dataset-cfg.non-overlapping 32 --shared-cfg.save-every 5 --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.lr 5e-4 --model-cfg.policy-cfg.pretrained-path /path/to/output_dir/pretrain_droid/checkpoint-3.pth
```

## Using Llama 2
We can also finetune from Llama2-7B. Please request access to the pre-trained [Llama 2](https://llama.meta.com/llama2/) from this form. In particular, we use llama-2-7b as the base model. Currently we do not have plans to support Llama 3, since reaching a near 15Hz control frequency using FlashAttention-2 is challenging without quantization. Please feel free to open a PR.
```bash
torchrun --nproc_per_node=8 --master_port=2450 scripts/train.py --dataset-cfg.dataset-json config/dataset_config.json --logging-cfg.output-dir /path/to/output_dir --logging-cfg.log-name finetune_icrt_llama2_7b --optimizer-cfg.warmup-epochs 1.25 --trainer-cfg.epochs 125 --model-cfg.vision-encoder-cfg.vision-encoder /path/to/vision/encoder.pth --dataset-cfg.num-repeat-traj 2 --model-cfg.policy-cfg.no-prompt-loss --model-cfg.policy-cfg.phase finetune --dataset-cfg.non-overlapping 32 --shared-cfg.save-every 5 --dataset-cfg.rebalance-tasks --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.lr 5e-4 --model-cfg.policy-cfg.lora-rank 32 --model-cfg.policy-cfg.llama-ckpt-dir /path/to/llama-2/llama-2-7b
```

## Easter Eggs
1. We also include the code for Gaussian Mixture Models (GMMs) or Diffusion Models as the prediction head for you to experiment with. 
2. The vision encoder can be unfrozen, which is controlled by the VisionEncoderConfig in [args](icrt/util/args.py). You can LoRA or entirely unfreeze the vision encoder. While in early experiments, unfreezing the vision encoder shows promises, finetuning the policy jointly with the vision encoder may take more than 2 day to train on 8xA100.
3. The key bottleneck for training ICRT is your disk accessing speed. A **high** read speed (i.e. all data on SSD or in [memory](https://superuser.com/questions/45342/when-should-i-use-dev-shm-and-when-should-i-use-tmp)) is necessary for the model to train under one day. 