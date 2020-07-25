
import torch.nn as nn
import torch
import newnet.config as conf
import newnet.net as net

class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, trans_fun, w_b, num_gs):
        proj_block = (w_in != w_out) or (stride != 1)
        if proj_block:
            h, w = cx["h"], cx["w"]
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx["h"], cx["w"] = h, w  # parallel branch
        cx = trans_fun.complexity(cx, w_in, w_out, stride, w_b, num_gs)
        return cx




class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = BasicTransform
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module(f"b{i + 1}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, w_b=None, num_gs=1):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_f = BasicTransform
            cx = ResBlock.complexity(cx, b_w_in, w_out, b_stride, trans_f, w_b, num_gs)
        return cx

class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self):
        super(ResNet, self).__init__()

        self._construct_cifar()
        self.apply(net.init_weights)

    def _construct_cifar(self):
        err_str = "Model depth should be of the format 6n + 2 for cifar"
        d = 8

        self.stem = ResStemCifar(3, 16)
        self.s1 = ResStage(16, 16, stride=1, d=d)
        self.s2 = ResStage(16, 32, stride=2, d=d)
        self.s3 = ResStage(32, 64, stride=2, d=d)
        self.head = ResHead(64, 10)

    def _construct_imagenet(self):
        g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)
        self.head = ResHead(2048, nc=10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        if "cifar" in cfg.TRAIN.DATASET:
            d = int((cfg.MODEL.DEPTH - 2) / 6)
            cx = ResStemCifar.complexity(cx, 3, 16)
            cx = ResStage.complexity(cx, 16, 16, stride=1, d=d)
            cx = ResStage.complexity(cx, 16, 32, stride=2, d=d)
            cx = ResStage.complexity(cx, 32, 64, stride=2, d=d)
            cx = ResHead.complexity(cx, 64, nc=cfg.MODEL.NUM_CLASSES)
        else:
            g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
            (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
            w_b = gw * g
            cx = ResStemIN.complexity(cx, 3, 64)
            cx = ResStage.complexity(cx, 64, 256, 1, d=d1, w_b=w_b, num_gs=g)
            cx = ResStage.complexity(cx, 256, 512, 2, d=d2, w_b=w_b * 2, num_gs=g)
            cx = ResStage.complexity(cx, 512, 1024, 2, d=d3, w_b=w_b * 4, num_gs=g)
            cx = ResStage.complexity(cx, 1024, 2048, 2, d=d4, w_b=w_b * 8, num_gs=g)
            cx = ResHead.complexity(cx, 2048, nc=cfg.MODEL.NUM_CLASSES)
        return cx

class ResHead(nn.Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx
class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in = 3, w_out = 10):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(w_out, 2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx

class BasicNet(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self):

        super(BasicNet, self).__init__()
        self.a = nn.Conv2d(3, 10, 3, stride=1, padding=1, bias=False)

        #self.b_bn.final_bn = True
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        x = self.a(x)
  
        x = self.avg_pool(x)
 
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out)
        self.a_relu = nn.ReLU()
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out)
        #self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx

if __name__ == '__main__':
    data = torch.rand(2,3,224,244)
    print(data.shape)
    #model = BasicTransform(3,6,1)
    model = ResNet()
    output = model(data)
    print(model)
    print(output.shape)
