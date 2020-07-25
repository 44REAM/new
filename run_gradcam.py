import sys
from dotenv import load_dotenv

sys.path.append("~/dream/newnet")
#from newnet import trainer
from newnet import gradcam
from newnet.config import cfg


load_dotenv()
cfg.merge_from_file("/home/yodchanan/dream/newnet/newnet/config/default.yaml")

#trainer.train_model()
gradcam.test()
