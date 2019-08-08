#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import logging
import numpy
import os

# Udl class
class Udl(object):
     # Constructor of UDL class takes a callback function
     # This function is expected to pack all of the necessary information into a single blob
     # for use by the Converter core.
     def __init__(self, layer_callback):
         self._layer_callback = layer_callback
         self._input_axes_order = []
         self._output_axes_order = []
         self._src_input_axes_order = []
         self._src_output_axes_order = []

     def getLayerCallback(self):
         return self._layer_callback

     # Optionally, this function could be used to specify UDL's expected
     # input and output axes order.
     #
     # By default, UDL's expected input/output axes order is for 3D input
     # { AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL } i.e HWC
     #
     # If UDL can handle multi-dimensional inputs i.e. 3D, 2D and 1D,
     # this function needs to be called for each input rank.
     def addAxisOrder(self, input_axes_order, output_axes_order):
         assert(isinstance (input_axes_order, list))
         assert(isinstance (output_axes_order, list))
         self._input_axes_order.append(input_axes_order)
         self._output_axes_order.append(output_axes_order)

     # Optionally, this function could be used to specify the corresponding src's
     # input and output axes order.
     #
     # By default, src's expected input/output axes order is
     # { AxisAnnotation.BATCH, AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH }
     #
     # If UDL can handle multi-dimensional inputs i.e. 3D, 2D and 1D,
     # this function needs to be called for each input rank.
     #
     # If addAxisOrder is called to specify axes order different than HWC, addSrcAxisOrder
     # needs to be called to specify the corresponding axes order.
     def addSrcAxisOrder(self, src_input_axes_order, src_output_axes_order):
         assert(isinstance (src_input_axes_order, list))
         assert(isinstance (src_output_axes_order, list))
         self._src_input_axes_order.append(src_input_axes_order)
         self._src_output_axes_order.append(src_output_axes_order)

     def getAxisOrder(self):
         return self._input_axes_order, self._output_axes_order

     def getSrcAxisOrder(self):
         return self._src_input_axes_order, self._src_output_axes_order

# default, empty blob
class UdlBlob(object):
     def __init__(self):
         self._blob = ''
         self._size = len(_blob)

     def getBlob(self):
         return self._blob

     def getSize(self):
         return self._size

class UdlBlobOutput(object):
     def __init__(self, blob, out_dims):
         self._blob = blob
         # _out_dims is a list of lists
         # a list where each index is a list of the dims
         self._out_dims = out_dims
         assert(isinstance (self._out_dims, list))
         for dims in self._out_dims:
              assert(isinstance (dims, list))

     def getBlob(self):
         return self._blob

     def getOutputDims(self, idx):
         return self._out_dims[idx]

