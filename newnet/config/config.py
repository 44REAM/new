from yacs.config import CfgNode as CN
import numpy as np
import random

_C = CN(new_allowed = True)
cfg = _C

_C.NUM_GPUS = 1

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.MODEL = 'anynet'
# The all important scales for the stuff
_C.TRAIN.IM_SIZE = 224
_C.TRAIN.BATCH_SIZE = 16

#--------------------------------
_C.DATASET = CN()
_C.DATASET.PATH = 'D:\\GoogleDrive\\dataset\\radiology\\tuberculosis_xray'
_C.DATASET.NAME = 'tb'
_C.DATASET.INPUT_FILTER = 1
_C.DATASET.SHUFFLE = True
_C.DATASET.N_CLASSES = 2

#--------------------------------

_C.OPTIM = CN()
# A very important hyperparameter
_C.OPTIM.BASE_LR = 0.01
_C.OPTIM.LR_POLICY ='cos'

_C.OPTIM.MAX_EPOCH = 1

_C.OPTIM.BURNIN = 0
_C.OPTIM.WARMUP_FACTOR = 0.1
# The all important scales for the stuff

_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0.9
_C.OPTIM.SGD.NESTEROV =True
_C.OPTIM.SGD.DAMPENING = 0.0
_C.OPTIM.SGD.WEIGHT_DECAY = 0.0001

#--------------------------------

_C.WANDB = CN()

_C.WANDB.ENTITY = "quacktab" 
_C.WANDB.PROJECT= 'default_project'
_C.WANDB.NAME = 'default_name'

#--------------------------------

_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True

#--------------------------------

_C.BN = CN()

# Perform ReLU inplace
_C.BN.EPS = 1e-5
_C.BN.MOMENTUM = 0.1

#--------------------------------
_C.ANYNET = CN()

_C.ANYNET.STEM_OUT = 6

_C.ANYNET.STAGE_LIMIT_MIN = 1
_C.ANYNET.STAGE_LIMIT_MAX = 1

_C.ANYNET.DEPTH_LIMIT = 4
_C.ANYNET.WIDTH_LIMIT = 4
_C.ANYNET.B_RATIO_LIMIT = [1]

_C.ANYNET.BIAS = False

#---------------------------------------- Randomly assign in set_config()

_C.ANYNET.STAGES = 1
_C.ANYNET.DEPTH_ARRAY = [1]
_C.ANYNET.WIDTH_ARRAY = [1]
_C.ANYNET.B_RATIO_ARRAY = [1]

#--------------------------------

_C.GRADCAM = CN()
_C.GRADCAM.BATCH_INDEX = 0
_C.GRADCAM.SHOW_TARGET = 0
_C.GRADCAM.NUM_GPUS = 1


#--------------------------------
_C.RANDOM_SEED = 1

def random_exp(limit):
    uni_limit = int(np.log(limit)/np.log(2))
    expo = np.random.randint(1,uni_limit)
    return 2**expo

def set_config():
    #---------------------------ANYNET----------------------------
    stages = np.random.randint(_C.ANYNET.STAGE_LIMIT_MIN,_C.ANYNET.STAGE_LIMIT_MAX+1)
    _C.ANYNET.STAGES = stages

    depth_array = []
    width_array=[]
    b_ratio_array=[]

    for i in range(_C.ANYNET.STAGES):

        depth = random_exp(_C.ANYNET.DEPTH_LIMIT)
        depth_array.append(depth)

        width = random_exp(_C.ANYNET.WIDTH_LIMIT)
        width_array.append(width)

        b_ratio = random.sample(_C.ANYNET.B_RATIO_LIMIT, 1)[0]
        b_ratio_array.append(b_ratio)

    _C.ANYNET.DEPTH_ARRAY = depth_array
    _C.ANYNET.WIDTH_ARRAY = width_array
    _C.ANYNET.B_RATIO_ARRAY = b_ratio_array

    _C.WANDB.NAME = f'S_{str(_C.ANYNET.STAGES)}'
  #---------------------------ANYNET----------------------------



if __name__ == '__main__':
    cfg.merge_from_file('../config/default.yaml')
    set_config()
    print(cfg.ANYNET)