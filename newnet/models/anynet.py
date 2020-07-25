
import torch 
import torch.nn as nn

from newnet.config import cfg

class AnyStem(nn.Module):

    def __init__(self,w_in,w_out ):
        super().__init__()
        self.conv = nn.Conv2d(w_in,w_out,3,stride=2, bias =cfg.ANYNET.BIAS)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.view = View()
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.view(x)
        x = self.fc(x)
        return x

class AnyBlock(nn.Module):
    def __init__(self, w_in, w_out, b_ratio, project = False):
        super().__init__()

        self.project = False if w_in==w_out else True

        if self.project:

            self.proj = nn.Conv2d(w_in, w_out, 1, bias=cfg.ANYNET.BIAS)
            self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)

        self.inv_bottleneck = BottleNeck(w_in, w_out,b_ratio)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):

        if self.project:
            x =  self.bn(self.proj(x)) + self.inv_bottleneck(x) 
        else:

            # CANNOT USE += ERRORRRR
            x = x+ self.inv_bottleneck(x)
        return x

class BottleNeck(nn.Module):

    def __init__(self,w_in, w_out,b_ratio):
        super().__init__()
        w_b = int(w_out*b_ratio)
        self.conv1 = nn.Conv2d(w_in, w_b, 3, padding = 1, bias = cfg.ANYNET.BIAS)
        self.bn1 = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
        self.relu1 = nn.ReLU(cfg.MEM.RELU_INPLACE)

        self.depthwise_conv=depthwise_conv(w_b)
        self.bn2 = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
        self.relu2 = nn.ReLU(cfg.MEM.RELU_INPLACE)

        self.conv2 = nn.Conv2d(w_b, w_out,3, padding = 1, bias = cfg.ANYNET.BIAS)
        self.bn3 = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOMENTUM)
        self.bn3.final_bn = True

    def forward(self,x):
       
        for layer in self.children():
            x = layer(x)
        return x

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out,  bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))

        self.a = nn.Conv2d(w_in, w_b, 1, padding=0, bias=cfg.ANYNET.BIAS)
        self.a_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        self.b = nn.Conv2d(w_b, w_b, 3, padding=1, groups=w_b, bias=cfg.ANYNET.BIAS)
        self.b_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=cfg.ANYNET.BIAS)
        self.c_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # ???
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


def depthwise_conv(w_in):
    return nn.Conv2d(w_in, w_in, 3,padding = 1, groups = w_in, bias = cfg.ANYNET.BIAS)

def pointwise_conv(w_in, w_out):
    return nn.Conv2d(w_in, w_out, 1, bias = cfg.ANYNET.BIAS)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, w_in,w_out):
        self.depthwise_conv = depthwise_conv(w_in)
        self.pointwise_conv = pointwise_conv(w_in,w_out)

    def forward(self,x):
        x = self.depthwise_conv(x)
        x=self.pointwise_conv(x)
        return x

class AnyStage(nn.Module):

    def __init__(self, depth,w_in, w_out, b_ratio):
        super().__init__()

        for i in range(depth):

            name = f"block_{i+1}"
            w_in = w_in if i==0 else w_out
            self.add_module(name, AnyBlock( w_in, w_out, b_ratio ))

    def forward(self,x):
        for layer in self.children():
  
            x = layer(x)
        return x

class AnyNet(nn.Module):

    def __init__(self,stem_out ,stages, depth_array, width_array,b_ratio_array, n_class):
        super().__init__()

        # define stem
        self.stem = AnyStem(cfg.DATASET.INPUT_FILTER,stem_out)
        w_in = stem_out

        # define N stages
        for i in range(stages):
            name = f"stage_{i+1}"
            self.add_module(name, AnyStage(depth_array[i],w_in ,width_array[i], b_ratio_array[i]))
            w_in = width_array[i]

        self.head = AnyHead(width_array[-1],n_class)

    def forward(self,x):
        for layer in self.children():
            x = layer(x)

        return x


if __name__ == "__main__":
    # stem = AnyStem(3, 6)

    # model = DepthwiseConv2d(3)

    # model = BottleNeck(3,6,5)
    # model = AnyBlock(3,5,2)
    # model = AnyStage(3,5,2,[10,6] )
    model = AnyNet(5, 3, [2,3,1],[5,4,3],[6,5,2],2)
    print(model)
    data = torch.rand(1,3,512,512)
    print(model(data).shape)