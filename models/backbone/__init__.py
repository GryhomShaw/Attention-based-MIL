from .resnet import resnet101
from .resnet import resnet50
from .mobilenetv2 import mobilenet_v2
from .vgg import vgg16, vgg19


class Backbone:
    model_zoo = {
        "resnet101": resnet101,
        "resnet50": resnet50,
        "mobilenetv2": mobilenet_v2,
        "vgg16": vgg16,
        "vgg19": vgg19
    }
