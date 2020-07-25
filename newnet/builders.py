#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from newnet.models import AnyNet
from newnet.config import cfg


# Supported models
_models = {"anynet": AnyNet}

def get_model():
    assert cfg.TRAIN.MODEL in _models.keys()
    
    if cfg.TRAIN.MODEL == 'anynet':
        model = _models[cfg.TRAIN.MODEL](cfg.ANYNET.STEM_OUT ,cfg.ANYNET.STAGES, cfg.ANYNET.DEPTH_ARRAY,cfg.ANYNET.WIDTH_ARRAY, 
            cfg.ANYNET.B_RATIO_ARRAY, cfg.DATASET.N_CLASSES)
        
    return model

