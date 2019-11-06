# -*- coding: utf-8 -*-
from models.semantic_segmentation.fcn import fcn,fcn8s,fcn16s,fcn32s
from models.semantic_segmentation.pspnet import pspnet
from easydict import EasyDict as edict

def get_segmentation_network(class_number):
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
    #fixed input shape, cannot change
    config.input_shape=[256,256]
    config.upsample_type='bilinear'
    config.use_bn=True
    config.use_dropout=False
    config.use_bias=False

    return globals()[config.net_name](config)