# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from math import ceil, floor

from .onnx_translations import *


# ------------------------------------------------------------------------------
#   AveragePool, MaxPool
# ------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('auto_pad', 's', ''),
                                    ('kernel_shape', 'li'),
                                    ('pads', 'li', [0, 0, 0, 0]),
                                    ('strides', 'li', [1, 1]))

        log_assert(pads_symmetric(params.pads) or pads_righthanded(params.pads),
                   code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))

        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name)
        if pads_righthanded(params.pads):
            padding_size_strategy = "PADDING_SIZE_EXPLICIT_ASYMMETRIC"
        if str(src_op.op_type) == 'AveragePool':
            pool_type = "POOL_AVG"
        else:
            pool_type = "POOL_MAX"

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_y=params.kernel_shape[0],
                                 size_x=params.kernel_shape[1],
                                 stride_y=params.strides[0],
                                 stride_x=params.strides[1],
                                 pad_y=params.pads[2],
                                 pad_x=params.pads[3],
                                 padding_size_strategy=padding_size_strategy,
                                 pool_region_include_padding=False)

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        input_height = input_shape[2]
        input_width = input_shape[3]
        output_height = self.calc_pool_output_dim(input_height,
                                                  op.size_y,
                                                  op.pad_y,
                                                  op.stride_y,
                                                  op.padding_size_strategy)
        output_width = self.calc_pool_output_dim(input_width,
                                                 op.size_x,
                                                 op.pad_x,
                                                 op.stride_x,
                                                 op.padding_size_strategy)
        output_shape = input_shape[0:2] + [output_height, output_width]
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, output_shape))
        return [output_shape]

    @staticmethod
    def calc_pool_output_dim(input_size, pool_size, padding, stride, padding_size_strategy):
        padding = -padding
        full_size = input_size - 2 * padding - pool_size

        if padding_size_strategy == "PADDING_SIZE_IMPLICIT_VALID":
            output_dim = ceil(1 + full_size) / stride
        elif padding_size_strategy == "PADDING_SIZE_IMPLICIT_SAME":
            output_dim = ceil(float(input_size) / stride)
        elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_FLOOR":
            output_dim = 1 + floor(full_size/stride)
        elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_ASYMMETRIC":
            # this is implemented for EXPLICIT_RIGHTHANDED in snpe c++ but modeltools maps
            # asymmetric to righthanded so mimicking that here
            full_size = input_size - padding - pool_size
            output_dim = 1 + floor(full_size / stride)
        else:  # EXPLICIT or UNDEFINED
            output_dim = 1 + ceil(full_size / stride)

        if (output_dim - 1) * stride + padding >= input_size:
            # don't start a pool beyond the right border of the image
            output_dim -= 1

        return int(output_dim)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.PoolOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxPoolTranslation(),
                                      onnx_type('AveragePool'),
                                      onnx_type('MaxPool'),
                                      op_adapter.PoolOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   BatchNormalization
# ------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('epsilon', 'f', 1e-5),
                                    ('is_test', 'i', 0),
                                    ('spatial', 'i', 1))

        # Extract op version to determine if we need to check if test or training
        # Starting version 7 onnx has dropped the is_test attribute
        version = OpVersionInfo.onnx_op_ver(src_op, self.get_supported_version())
        if version < 7:
            log_assert(params.is_test, code_to_message.get_error_message("ERROR_BATCHNORM_TEST_ONLY"))

        input_names = list(src_op.input)
        gamma, beta, mu, var = graph.weights.fetch(*input_names[1:])
        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        weights = gamma/numpy.sqrt(var+params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        bias = -mu*weights + beta

        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      across_spatial=bool(params.spatial))

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.BatchnormOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxBatchNormalizationTranslation(),
                                      onnx_type('BatchNormalization'),
                                      op_adapter.BatchnormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))

        weights = graph.weights.fetch(input_names[1])

        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[0], dtype=numpy.float32)

        params = extract_attributes(src_op,
                                    ('auto_pad', 's', ''),
                                    ('dilations', 'li', [1, 1]),
                                    ('group', 'i', 1),
                                    ('kernel_shape', 'li', []),
                                    ('pads', 'li', [0, 0, 0, 0]),
                                    ('strides', 'li', [1, 1]))

        log_assert(pads_symmetric(params.pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))

        if params.kernel_shape:
            log_assert(tuple(params.kernel_shape) == weights.shape[2:],
                       code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))

        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)

        return op_adapter.ConvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        padx=params.pads[1],
                                        pady=params.pads[0],
                                        padding_size_strategy=padding_mode,
                                        stridex=params.strides[1],
                                        stridey=params.strides[0],
                                        dilationx=params.dilations[1],
                                        dilationy=params.dilations[0],
                                        groups=params.group)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def infer_output_shapes(self, op, input_shapes):
        input_height = input_shapes[0][2]
        input_width = input_shapes[0][3]

        output_height = self.calc_conv_output_dim(input_height,
                                                  op.weights.shape[2],
                                                  op.pady,
                                                  op.stridey,
                                                  op.dilationy,
                                                  op.padding_size_strategy)
        output_width = self.calc_conv_output_dim(input_width,
                                                 op.weights.shape[3],
                                                 op.padx,
                                                 op.stridex,
                                                 op.dilationx,
                                                 op.padding_size_strategy)
        output_depth = op.bias.shape[0]
        batch = input_shapes[0][0]
        output_shape = [batch, output_depth, output_height, output_width]
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, output_shape))
        return [output_shape]

    @staticmethod
    def calc_conv_output_dim(input_size, filter_size, padding, stride, dilation, padding_size_strategy):

        kernel_extent = dilation * (filter_size - 1) + 1
        full_size = float(2 * padding) + input_size - kernel_extent

        if padding_size_strategy == "PADDING_SIZE_IMPLICIT_VALID":
            filter_ = int(filter_size + ((filter_size - 1) * (dilation - 1)))
            output_dim = ceil(float(input_size - filter_ + 1) / float(stride))
        elif padding_size_strategy == "PADDING_SIZE_IMPLICIT_SAME":
            output_dim = ceil(float(input_size) / float(stride))
        elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_FLOOR":
            output_dim = 1 + floor(full_size/float(stride))
        else:  # EXPLICIT or UNDEFINED
            output_dim = 1 + (full_size / float(stride))

        return int(output_dim)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ConvolutionOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxConvTranslation(),
                                      onnx_type('Conv'),
                                      op_adapter.ConvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConvTranspose
# ------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        weights = graph.weights.fetch(input_names[1])
        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[1], dtype=numpy.float32)  # take the second dim because Onnx weights for
                                                                       # convtranspose is CMHW

        params = extract_attributes(src_op,
                                    ('auto_pad', 's', ''),
                                    ('dilations', 'li', [1, 1]),
                                    ('group', 'i', 1),
                                    ('kernel_shape', 'li', []),
                                    ('output_padding', 'li', []),
                                    ('output_shape', 'li', [0, 0]),
                                    ('pads', 'li', [0, 0, 0, 0]),
                                    ('strides', 'li', [1, 1, ]))

        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)
        log_assert(pads_symmetric(params.pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))
        if params.kernel_shape:
            log_assert(tuple(params.kernel_shape) == weights.shape[2:],
                       code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))

        log_assert(params.strides[0] == params.strides[1],
                   code_to_message.get_error_message("ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED"))

        op = op_adapter.DeconvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        stride=params.strides[0],
                                        padding=params.pads[0],
                                        padding_size_strategy=padding_mode,
                                        output_height=params.output_shape[0],
                                        output_width=params.output_shape[1],
                                        groups=params.group)

        log_assert(not params.output_padding,
                   code_to_message.get_error_message("ERROR_DECONV_OUTPUT_PADDING_UNSUPPORTED"))
        return op

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        if op.output_height == 0:
            # calculate according to provided formula
            input_height = input_shape[2]
            input_width = input_shape[3]

            def calc_output_dim(input_size,
                                filter_size,
                                stride,
                                pad):
                return stride*(input_size-1) + filter_size - 2*pad  # + output_pad

            output_height = calc_output_dim(input_height,
                                            op.weights.shape[2],
                                            op.stride,
                                            op.padding)
            op['output_height'] = output_height

            output_width = calc_output_dim(input_width,
                                           op.weights.shape[3],
                                           op.stride,
                                           op.padding)
            op['output_width'] = output_width
        else:
            output_height = op.output_height
            output_width = op.output_width

        return [input_shape[0:2] + [output_height, output_width]]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.DeconvolutionOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxConvTransposeTranslation(),
                                      onnx_type('ConvTranspose'),
                                      op_adapter.DeconvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   FC
# ------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis', 'i', 1),
                                    ('axis_w', 'i', 1))
        log_assert(params.axis == 1, code_to_message.get_error_message("ERROR_FC_AXIS_UNSUPPORTED"))
        log_assert(params.axis == 1, code_to_message.get_error_message("ERROR_FC_AXIS_W_UNSUPPORTED"))

        input_names = graph.get_input_names(src_op)
        weights, bias = graph.weights.fetch(*input_names[1:3])
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        N = op.weights_list[0].shape[1]
        M = input_shapes[0][0]
        return [[M, N]]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.FullyConnectedOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxFCTranslation(),
                                      onnx_type('FC'),
                                      op_adapter.FullyConnectedOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
# ------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            pool_type = "POOL_AVG"
        else:
            pool_type = "POOL_MAX"

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_x=input_buf.shape[3],
                                 size_y=input_buf.shape[2],
                                 stride_x=input_buf.shape[3],
                                 stride_y=input_buf.shape[2])

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.PoolOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxGlobalPoolTranslation(),
                                      onnx_type('GlobalAveragePool'),
                                      onnx_type('GlobalMaxPool'))


# ------------------------------------------------------------------------
#   InstanceNormalization
# ------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      compute_statistics=True,
                                      use_mu_sigma=True,
                                      across_spatial=True)
    # rest is handled by OnnxBatchNormalizationTranslation


# ------------------------------------------------------------------------------
#   MaxRoiPool
# ------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('spatial_scale', 'f', 1.0),
                                    ('pooled_shape', 'li'))

        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        roi_buf = graph.get_buffer(input_names[1])
        output_shape = [ roi_buf.shape[0],
                         input_buf.shape[1],
                         params.pooled_shape[0],
                         params.pooled_shape[1] ]

        return op_adapter.RoiPoolingOp(src_op.name,
                                       output_shape,
                                       pooled_size_h=params.pooled_shape[0],
                                       pooled_size_w=params.pooled_shape[1],
                                       spatial_scale=params.spatial_scale)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.RoiPoolingOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxMaxRoiPoolTranslation(),
                                      onnx_type('MaxRoiPool'),
                                      op_adapter.RoiPoolingOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Prelu, LeakyRelu
# ------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])

        if str(src_op.op_type) == 'LeakyRelu':
            params = extract_attributes(src_op, ('alpha', 'f', 0.01))
            bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
            bias *= params.alpha
        else:
            slope = graph.weights.fetch(input_names[1])
            if len(slope) == 1:
                bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
                bias *= slope[0]
            else:
                bias = numpy.require(slope, dtype=numpy.float32)

        return op_adapter.PreluOp(src_op.name, bias)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.PreluOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxPreluTranslation(),
                                      onnx_type('Prelu'),
                                      onnx_type('LeakyRelu'),
                                      op_adapter.PreluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Lrn
# ------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('alpha', 'f'),
                                    ('beta', 'f'),
                                    ('bias', 'f', 1.0),
                                    ('size', 'i'))

        return op_adapter.RNormOp(src_op.name,
                                  params.size,
                                  params.alpha / params.size,
                                  params.beta,
                                  params.bias,
                                  across_channels=True)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.RNormOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxLrnTranslation(),
                                      onnx_type('LRN'),
                                      op_adapter.RNormOp.TRANSLATION_KEY)

