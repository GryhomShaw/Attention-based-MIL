from .densenet_cam import densenet121
from .resnet_sim import resnet101 as resnet101_sim
from .resnet import resnet101
from .resnet import resnet50
from .mobilenetv2 import mobilenet_v2
from .vgg import vgg16, vgg19


class Backbone:
    model_zoo = {
        "densenet121": densenet121(pretrained=False, num_classes=2),
        "resnet101_sim": resnet101_sim(pretrained=False, num_classes=2),
        "resnet101": resnet101(pretrained=True),
        "resnet50": resnet50(pretrained=True),
        "mobilenetv2_05": mobilenet_v2(pretrained=False, width_mult=0.5, num_classes=2),
        "mobilenetv2_10": mobilenet_v2(pretrained=True),
        "vgg16": vgg16(pretrained=True),
        "vgg19": vgg19(pretrained=True)
    }
