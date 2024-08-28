import math
import sys
from typing import Iterable, Union

import torch
import torch.nn as nn
from . import misc, lr_sched

from icrt.util.args import ExperimentConfig
from icrt.models.policy import ICRT

def train_one_epoch(model: ICRT, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, validate=False,
                    args : ExperimentConfig=None):
    if validate:
        model.eval()
    else:
        model.train()
        optimizer.zero_grad() # Clear gradients only during training

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.trainer_cfg.accum_iter

    # breakpoint()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, dataset_item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        for k, v in dataset_item.items():
            dataset_item[k] = v.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_dict = model(dataset_item)
        
        loss_value = loss.item()
        loss_value_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_value_dict = {k: v / accum_iter for k, v in loss_value_dict.items()}
        if not validate:
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0 and not validate:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_dict_reduce = {k: misc.all_reduce_mean(v) for k, v in loss_value_dict.items()}
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if not validate:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('train_{}'.format(k), v, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
            else:
                log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('val_{}'.format(k), v, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
