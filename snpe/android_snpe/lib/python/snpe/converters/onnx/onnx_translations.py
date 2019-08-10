# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import translation, op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from .util import *

OP_VERSION_SUPPORTED = {
    'argmax': [1],
    'input': [1],
    'batchnorm': [1, 6, 7],
    'convolution': [1],
    'concatenation': [1, 4],
    'constant': [1],
    'crop': [1],
    'deconvolution': [1],
    'elementwise_max': [1, 6, 8],
    'elementwise_product': [1, 6, 7],
    'elementwise_sum': [1, 6, 7],
    'fully_connected': [1],  # Handles FC, GEMM and MatMul. Ignored GEMM op set until it's support is there.
    'gather': [1],
    'gru': [1, 7],
    'lstm': [1, 7],
    'neuron': [1, 6],  # Handles Clip, Relu, Sigmoid, Tanh , and Elu operations for now.
    'pad': [1, 2],
    'pool': [1],
    'permute': [1],
    'prelu': [1, 6, 7],
    'reshape': [1, 5],  # Handles Flatten {1} and Reshape {1, 5} operations. Used the larger set for now.
    'rnorm': [1],
    'roi_pooling': [1],
    'rnn': [1],
    'resize': [1],  # TO_DO
    'shape': [1],
    'slice': [1],
    'squeeze': [1],
    'softmax': [1],
    'unsqueeze': [1]
}

OnnxTranslations = translation.TranslationBank()

# onnx specific translation method keys
ADD_INPUT_OP = "ADD_INPUT_OP"
SUPPORTED_VERSION = "SUPPORTED_VERSION"


class OnnxTranslationBase(translation.ConversionTranslationBase):
    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)
        self.register_method(SUPPORTED_VERSION, self.get_supported_version)

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.output))

    def populate_axes_format(self, node, graph):
        output_buffers = graph.get_output_buffers(node)
        for buf in output_buffers:
            if node.op.type == op_adapter.InputOp.TRANSLATION_KEY:
                if node.op.image_type == 'opaque':
                    buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                elif buf.rank() == 4:
                    buf.axis_format = AxisTracker.AxisFormat.NCS
                    node.op.shape = buf.shape
                elif buf.rank() == 3:
                    buf.axis_format = AxisTracker.AxisFormat.TBF
                    node.op.shape = buf.shape
                elif buf.rank() == 2:
                    buf.axis_format = AxisTracker.AxisFormat.FEATURE
                    node.op.shape = buf.shape
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_INPUT_UNEXPECTED_RANK")(node.op.name,
                                                                                                      buf.rank()))
            else:
                if buf.rank() == 4:
                    buf.axis_format = AxisTracker.AxisFormat.NCS
                elif buf.rank() == 3:
                    buf.axis_format = AxisTracker.AxisFormat.TBF
                elif buf.rank() == 2:
                    buf.axis_format = AxisTracker.AxisFormat.FEATURE
                else:
                    buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def get_supported_version(self):
        raise NotImplementedError("get_supported_version for {} not implemented ".format(str(self.__class__.__name__)))


# -----------------------------------------------------------------
# Converter translations
# Note: ONNX doesn't have input op(s) but we create one for the IR
# -----------------------------------------------------------------
class OnnxInputTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_method(ADD_INPUT_OP, self.add_input_op)

    def add_input_op(self, input_, graph):
        name = str(input_.name)
        tensor_shape = input_.type.tensor_type.shape
        shape = [int(dim.dim_value) for dim in tensor_shape.dim]
        #TODO Handle proper image encoding conversions
        if len(shape) == 4:
            node = graph.add_input(name, shape, 'bgr', 'default')
        else:
            node = graph.add_input(name, shape, 'bgr', 'opaque')
        self.populate_axes_format(node, graph)



OnnxTranslations.register_translation(OnnxInputTranslation(),
                                      onnx_type('input'),
                                      op_adapter.InputOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Dropout, and other Noops
# ------------------------------------------------------------------------------
class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        return op_adapter.Noop(src_op.name)

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]

    def get_supported_version(self):
        return {}


OnnxTranslations.register_translation(OnnxNoopTranslation(),
                                      onnx_type('Dropout'),
                                      op_adapter.Noop.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   StaticOp
# ------------------------------------------------------------------------------
# 'Static' ops are transformations applied to weights, which do not produce
# an actual runtime output.
class OnnxStaticTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        return op_adapter.StaticOp(src_op.name)

    def infer_output_shapes(self, op, input_shapes):
        return []

    def get_supported_version(self):
        return {}


OnnxTranslations.register_translation(OnnxStaticTranslation(), op_adapter.StaticOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Class OpVersionInfo
# ------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo:
    model_opset_version = 0

    def __init__(self):
        self.op_version_dict = dict()
        self.setup_op_version_dict()

    def setup_op_version_dict(self):
        for schema in defs.get_all_schemas_with_history():
            # Splitting the operator name and storing the version in op_version_dict
            self.op_version_dict[op_type(schema.name)] = schema.since_version

    def get_op_ver_dict(self):
        return self.op_version_dict

    def validate_op_ver(self, src_op, supported_version):
        if self.op_version_dict[op_type(src_op.op_type)] not in supported_version:
            log_warning(code_to_message.get_warning_message("WARNING_OP_NOT_SUPPORTED")(src_op.op_type))

    def set_global_op_ver(self, model):
        """ Sets the highest global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            log_warning(code_to_message.get_warning_message("WARNING_OPSET_VERSION"))

        for opset in model.opset_import:
            if opset.version > OpVersionInfo.model_opset_version:
                OpVersionInfo.model_opset_version = opset.version

    @staticmethod
    def onnx_op_ver(src_op, supported_version):
        """Return the actual op version. If embedded in the op name return that,
           otherwise get the global op version and correlate to the highest op version
           supported as per the onnx.proto specification"""
        onnx_data = get_op_info(src_op.op_type)
        # If op is missing version, use the version as the minimum of the supported
        # model opset version and the largest supported op version in the converter
        # TODO See if there is a way to lookup the current op version information for
        # a given model_opset_version... this is really what we should be using instead
        # of the actual model_opset_verison
        if onnx_data[1] == 0:
            min_supported_version = min(supported_version[-1], OpVersionInfo.model_opset_version)
            return min_supported_version
        return onnx_data[1]
