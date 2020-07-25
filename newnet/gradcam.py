import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import newnet.logger as logger
import newnet.datasets.loader as loader
from newnet.models import AnyNet
from newnet.config import cfg, set_config
from newnet.datasets import TBDatasets
from newnet.builders import get_model


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelIter():
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        
        for name, module in self.model._modules.items():
            
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            else:
                x =module(x)

        return target_activations, x

class GradCAM():
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.model.eval()

        self.feature_module = feature_module
        self.extractor  = ModelIter(self.model, self.feature_module, target_layers)

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):

        features, output = self.extractor(x)
        
        index = output.argmax(dim = 1).cpu().numpy()[0]
        
        one_hot = torch.zeros_like(output).cuda()
        one_hot[:,index] = 1
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[cfg.GRADCAM.SHOW_TARGET].cpu().numpy()

        target = features[cfg.GRADCAM.SHOW_TARGET]
        target = target.cpu().detach().numpy()[cfg.GRADCAM.BATCH_INDEX]

        target = target * grads_val.mean(axis = (2,3)).reshape(-1,1,1)
        cam = np.sum(target, axis = 0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, x.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
        
    
def setup_model():
    model = get_model()

    print(model)
    

    err_str = f"Avaliable gpu: {torch.cuda.device_count()}"
    assert cfg.GRADCAM.NUM_GPUS <= torch.cuda.device_count(), err_str

    cur_device = torch.cuda.current_device()

    model = model.cuda(device = cur_device)
    
    # use multiple GPU
    if cfg.GRADCAM.NUM_GPUS > 1:
        model = torch.nn.DataParallel(
            module=model, device_ids=[i for i in range(cfg.GRADCAM.NUM_GPUS)]
        )

    return model

def get_image_from_tensor(tensor):
    image = tensor[0].cpu().numpy()
    if image.shape[0] == 1:
        # return image shape (512,512)
        return image[0]

    # return image shape (512,512,3)
    return image

def test():
    # set initial config
    set_config()

    # load config from wandb
    logger.load_last_wandb_config()

    # load last state dict from wandb
    model = setup_model()
    ckpt = logger.load_last_wandb_checkpoint()
    
    # load state dict
    ms = model.module if cfg.GRADCAM.NUM_GPUS > 1 else model
    ms.load_state_dict(ckpt['model_state'])

    # get test data
    gradcam_loader = loader.construct_gradcam_loader()
    
    # seletect module where you want gradcam
    feature_module = model.module.stem if cfg.GRADCAM.NUM_GPUS > 1 else model.stem
    print(feature_module)
    gradcam = GradCAM(model, feature_module, ["conv"])

    for cur_iter, (inputs, labels) in enumerate(gradcam_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        mask = gradcam(inputs)
        img = get_image_from_tensor(inputs)

        show_cam_on_image(img,mask)

def show_cam_on_image(img, mask):
    plt.imshow(img,cmap='gray')
    heatmap = sns.heatmap(mask, alpha = 0.1,zorder = 2,cmap = 'coolwarm')
    heatmap.imshow(img, zorder = 1,cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ =="__main__":
    
    cfg.merge_from_file("/home/yodchanan/dream/newnet/newnet/config/test.yaml")
    test()