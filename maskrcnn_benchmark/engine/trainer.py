# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import numpy as np

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

####### add on 2019/01/10 ########
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.validation import EvalMetric
from maskrcnn_benchmark.utils.earlystop import EarlyStopping
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from ..utils.comm import is_main_process,get_rank
##################################

# add on 2019/01/10
def clean_model_cache(output_dir):
    filenms=os.listdir(output_dir)
    for file in filenms:
        if 'model' in file and 'final' not in file:
            os.remove(output_dir+file)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg=None,
    data_loader_val=None,
    whs=None,
    masksgt=None,
):
    
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"] # 0 at start
    
    output_dir=cfg.OUTPUT_DIR # used for deleting the outdated model to save the memory
    
    
    ##########################################
    ## early stop ##
    if cfg.PATIENCE and data_loader_val:
        early_stopping = EarlyStopping(patience=cfg.PATIENCE, verbose=True)
    ##########################################
    
    model.train() # begin training
    start_training_time = time.time() 
    end = time.time()
    
    ######################################################################################################################
    # Note: once begin training, rank of GPU will change iteratively! So the rank of GPU is 0 above this comment.
    #       Moreover, iteration is counted following rank 0 GPU. In other words, all checkpoint work on cuda:0.
    ######################################################################################################################
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        
        # add ignore 
        if len(targets[0]) < 1:
            print('num_boxes: ', len(targets[0]))
            continue
        
        iteration = iteration + 1
        arguments["iteration"] = iteration
       
        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # give all the information
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            clean_model_cache(output_dir)
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            ################## add evaluation here ##################### (2019/01/10)
            if cfg.DATASETS.VAL:
                iou_types = ("bbox",)
                if cfg.MODEL.MASK_ON:
                    iou_types = iou_types + ("segm",)
                    dataset_name = cfg.DATASETS.VAL
                if len(dataset_name)>1: # TODO: add support for multiple datasets
                    raise ValueError('Only support the single validation set, but get {}!'.format(len(dataset_name)))
                predictions=inference(
                                        model,
                                        data_loader_val,
                                        dataset_name=dataset_name,
                                        iou_types=iou_types,
                                        box_only=cfg.MODEL.RPN_ONLY,
                                        device=cfg.MODEL.DEVICE,
                                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                        only_predictions=True
                                    )
                model.train() # reset training
                synchronize() # 同步所有过程
                
                if is_main_process():
                    # compute dice coefficient
                    masker=Masker(cfg.EVAL_THRESHOLD)
                    print('Loading the predictions...')
                    masksdt=[]
                    for i,prediction in enumerate(predictions):
                        prediction=prediction.resize(whs[i])
                        maskdt=prediction.get_field('mask')
                        if list(maskdt.shape[-2:]) != list(whs[i][::-1]):
                            maskdt = masker(maskdt.expand(1, -1, -1, -1, -1), prediction)
                            maskdt = maskdt[0]
                        maskdt=maskdt.numpy().sum((0,1))
                        maskdt=(maskdt>0).astype(np.uint8)
                        masksdt.append(maskdt)
                    print('Loading Complete!')
                    mean_dice=EvalMetric(masksgt,masksdt).mean_dice
                    #print('The shape of gt and dt are: {} and {}'.format(masksgt[0].shape,masksdt[0].shape))
                    print('The length of gt and dt are: {} and {}'.format(len(masksgt),len(masksdt)))
                    logger.info('The mean dice coefficient: {}'.format(mean_dice))
                
                    ## early stop ##
                    if 'early_stopping' in dir():
                        print('Logging early stopping...')
                        early_stopping(mean_dice,logger)
                        if early_stopping.early_stop:
                            logger.info('Early stopping')
                            clean_model_cache(output_dir)
                            checkpointer.save("model_final", **arguments)
                            break
            #############################################################
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


# def do_train(
#     model,
#     data_loader,
#     optimizer,
#     scheduler,
#     checkpointer,
#     device,
#     checkpoint_period,
#     arguments,
# ):
#     logger = logging.getLogger("maskrcnn_benchmark.trainer")
#     logger.info("Start training")
#     meters = MetricLogger(delimiter="  ")
#     max_iter = len(data_loader)
#     start_iter = arguments["iteration"] # 0 at start
#     model.train() # begin training
#     start_training_time = time.time() 
#     end = time.time()
#     for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
#         data_time = time.time() - end
        
#         # add ignore 
#         if len(targets[0]) < 1:
#             print('num_boxes: ', len(targets[0]))
#             continue
        
#         iteration = iteration + 1
#         arguments["iteration"] = iteration
       
        
#         scheduler.step()

#         images = images.to(device)
#         targets = [target.to(device) for target in targets]

#         loss_dict = model(images, targets)

#         losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = reduce_loss_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#         meters.update(loss=losses_reduced, **loss_dict_reduced)

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         batch_time = time.time() - end
#         end = time.time()
#         meters.update(time=batch_time, data=data_time)

#         eta_seconds = meters.time.global_avg * (max_iter - iteration)
#         eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
#         # give all the information
#         if iteration % 20 == 0 or iteration == max_iter:
#             logger.info(
#                 meters.delimiter.join(
#                     [
#                         "eta: {eta}",
#                         "iter: {iter}",
#                         "{meters}",
#                         "lr: {lr:.6f}",
#                         "max mem: {memory:.0f}",
#                     ]
#                 ).format(
#                     eta=eta_string,
#                     iter=iteration,
#                     meters=str(meters),
#                     lr=optimizer.param_groups[0]["lr"],
#                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
#                 )
#             )
#         if iteration % checkpoint_period == 0:
#             checkpointer.save("model_{:07d}".format(iteration), **arguments)
#             # add validation inference here
#         if iteration == max_iter:
#             checkpointer.save("model_final", **arguments)

#     total_training_time = time.time() - start_training_time
#     total_time_str = str(datetime.timedelta(seconds=total_training_time))
#     logger.info(
#         "Total training time: {} ({:.4f} s / it)".format(
#             total_time_str, total_training_time / (max_iter)
#         )
#     )
