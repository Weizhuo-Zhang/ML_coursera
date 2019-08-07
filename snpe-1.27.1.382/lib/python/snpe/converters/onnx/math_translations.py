# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseSumOp(str(src_op.name))

        # check if Bias is provided as one of the inputs. This is determined if one
        # of the inputs is a const
        bias_input, actual_inputs = parse_out_weights_biases_inputs(src_op, graph)
        self.input_names = actual_inputs
        if len(bias_input):
            log_assert(len(bias_input) == 1, code_to_message.get_error_message("ERROR_MULTIPLE_CONST_INPUTS_FOUND")
                       (src_op.op_type, bias_input))
            op.bias = graph.weights.fetch(bias_input[0])

        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseSumOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      onnx_type('Add'),
                                      op_adapter.ElementwiseSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class OnnxArgMaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ArgMax
        params = extract_attributes(src_op,
                                    ('axis', 'i', 0),
                                    ('keepdims', 'i', True))

        return op_adapter.ArgMaxOp(src_op.name,
                                   axis=params.axis,
                                   keepdims=params.keepdims)

    def infer_output_shapes(self, op, input_shapes):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == op.axis:
                if op.keepdims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ArgMaxOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxArgMaxTranslation(),
                                      onnx_type('ArgMax'),
                                      op_adapter.ArgMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # these parameters belong to Elu
        params = extract_attributes(src_op,
                                    ('alpha', 'f', 1.0))
        return op_adapter.NeuronOp(src_op.name,
                                   "NEURON_ELU",
                                   a=params.alpha)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      onnx_type('Elu'))


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        params = extract_attributes(src_op,
                                    ('alpha', 'f', 1.0),
                                    ('beta', 'f', 1.0),
                                    ('transA', 'i', 0),
                                    ('transB', 'i', 0))

        log_assert(not params.transA,
                   code_to_message.get_error_message("ERROR_GEMM_TRANSPOSE_NOT_SUPPORTED"))
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        weights *= params.alpha
        # for GEMM, weights are supposed to be B and thus KxN.
        # for FC, weights are supposed to be NxK and get transposed
        # implicitly. Transpose explicitly here so that they wind up as NxK
        # for axes_to_snpe_order
        weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
        bias *= params.beta
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.FullyConnectedOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxGemmTranslation(), onnx_type('Gemm'))


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_MATMUL"))
        input_names = list(map(str, src_op.input))
        # SNPE currently only supports FC, so given AxB, B MUST be a set of
        # static weights
        weights = graph.weights.fetch(input_names[1])
        bias = numpy.zeros(weights.shape[1], dtype=numpy.float32)
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.FullyConnectedOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxMatMulTranslation(), onnx_type('MatMul'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        assert_no_broadcast(src_op)
        return op_adapter.ElementwiseMaxOp(str(src_op.name))

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseMaxOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      onnx_type('Max'),
                                      op_adapter.ElementwiseMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseProductOp(src_op.name)

        # check if Weights is provided as one of the inputs. This is determined if one
        # of the inputs is a const
        weight_input, actual_inputs = parse_out_weights_biases_inputs(src_op, graph)
        self.input_names = actual_inputs
        if len(weight_input):
            log_assert(len(weight_input) == 1, code_to_message.get_error_message("ERROR_MULTIPLE_CONST_INPUTS_FOUND")
                       (src_op.op_type, weight_input))
            op.weights = graph.weights.fetch(weight_input[0])

        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseProductOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      onnx_type('Mul'),
                                      op_adapter.ElementwiseProductOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(src_op.name, "NEURON_RELU")

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      onnx_type('Relu'),
                                      op_adapter.NeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(src_op.name, "NEURON_LOGISTIC", a=1.0)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), onnx_type('Sigmoid'))


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('axis', 'i', 1))
        input_buf = graph.get_buffer(str(src_op.input[0]))
        log_assert(params.axis == 1,
                   "Node %s: SNPE supports softmax only for axis 1",
                   src_op.name)
        log_assert(input_buf.rank() == 2,
                   "Node %s: SNPE supports softmax only for inputs of rank 2",
                   src_op.name)
        return op_adapter.SoftmaxOp(src_op.name)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.SoftmaxOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      onnx_type('Softmax'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseSumOp(src_op.name)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ElementwiseSumOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxSumTranslation(), onnx_type('Sum'))


# ------------------------------------------------------------------------------
#   Tanh, ScaledTanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op,
                                    ('alpha', 'f', 1.0),
                                    ('beta', 'f', 1.0))
        return op_adapter.NeuronOp(src_op.name,
                                   "NEURON_TANH",
                                   a=params.alpha,
                                   b=params.beta)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      onnx_type('Tanh'),
                                      onnx_type('ScaledTanh'))
