#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import struct

class LayerType:
    MY_CUSTOM_SCALE_LAYER = 1
    MY_ANOTHER_LAYER = 2


class MyCustomScaleLayerParam:
    def __init__(self):
        self.type = LayerType.MY_CUSTOM_SCALE_LAYER
        self.bias_term = None
        self.weights_dim = []
        self.weights_data = []

    def Serialize(self):
        packed = struct.pack('i', self.type)
        packed += struct.pack('?', self.bias_term)
        packed += struct.pack('I%sI' % len(self.weights_dim),
                              len(self.weights_dim), *self.weights_dim)
        packed += struct.pack('I%sf' % len(self.weights_data),
                              len(self.weights_data), *self.weights_data)
        return packed
