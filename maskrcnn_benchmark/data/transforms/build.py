# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

# update
# 2019/01/10: cancel flip augmentation, add color jitter and rotation

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN #0.5
        bright_lb = cfg.INPUT.BRIGHT_LB_TRAIN #0.5
        contrast_lb = cfg.INPUT.CONTRAST_LB_TRAIN #0.5
        jitter_prob = cfg.INPUT.JITTER_PROB_TRAIN # 0.5
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        bright_lb = 1
        contrast_lb = 1
        jitter_prob = 0.5

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    
    # Note: all the transformations here are self-defined in transformation.py
    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            #T.RandomHorizontalFlip(flip_prob), # we don't us Filp here
            T.ColorJitter(bright_lb,contrast_lb,jitter_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
