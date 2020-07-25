import sys

sys.path.append("~/dream/newnet")
from newnet import trainer
#from newnet import gradcam
from newnet.config import cfg


cfg.merge_from_file("/home/yodchanan/dream/newnet/newnet/config/cifar10.yaml")

trainer.train_model()
#gradcam.test()


