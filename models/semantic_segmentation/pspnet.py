# -*- coding: utf-8 -*-

import torch.nn as TN
from models.semantic_segmentation.backbone import backbone
from models.semantic_segmentation.modules import get_midnet, get_suffix_net

class pspnet(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name=self.__class__.__name__

        self.backbone = backbone(config)

        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
#        self.midnet_out_channels=self.config.midnet_out_channels
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        self.decoder = get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        x = self.decoder(feature_mid)
        self.center_feature=self.decoder.center_feature
        return x
