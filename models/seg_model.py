# -*- coding: utf-8 -*-
from models.semantic_segmentaiton.fcn import fcn,fcn8s,fcn16s,fcn32s
from models.semantic_segmentation.pspnet import pspnet
from easydict import EasyDict as edict

def get_segmentation_model(class_number):
    config=edict()
    config.use_none_layer=False
    config.backbone_name='vgg13'
    config.class_number=class_number
    config.upsample_layer=3
    config.backbone_freeze=False
    config.freeze_layer=2
    config.freeze_ratio=0
    config.backbone_pretrained=True
    config.layer_preference='last'
    config.net_name='fcn'

    return globals()[config.net_name](config)