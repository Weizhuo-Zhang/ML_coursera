# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from operator import mul
from functools import reduce

from snpe.converters.common.converter_ir import translation, op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.common.converter_ir.op_graph import BufferCriteria
from snpe.converters.common.utils.snpe_converter_utils import *
from snpe.converters.common.utils import code_to_message

# ------------------------------
#   Module Level enum/Functions
# ------------------------------
REMOVE_NOOP = "REMOVE_NOOP"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
FOLD_CONCATS = "FOLD_CONCATS"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
supported_opt_list = [SQUASH_SCALE, SQUASH_BATCHNORM, FOLD_CONCATS, MATCH_CHANNELSHUFFLE, AXES_TO_SPATIAL_FIRST_ORDER,
                      REMOVE_NOOP]

OptimizationTranslations = translation.TranslationBank()


class OptimizationTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


def apply_graph_optimizations(graph, disable_batchnorm_folding=False, **kwargs):

    # apply graph transformations
    log_debug2("Applying graph Optimizations...")
    OptimizationTranslations.apply_method_to_graph(SQUASH_SCALE, graph, fail_if_no_method=False)
    OptimizationTranslations.apply_method_to_graph(FOLD_CONCATS, graph, fail_if_no_method=False)
    OptimizationTranslations.apply_method_to_graph(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
    if not disable_batchnorm_folding:
        OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)

    # transition to NSC
    perform_axes_to_spatial_first_order = kwargs.get('perform_axes_to_spatial_first_order', True)
    if perform_axes_to_spatial_first_order:
        OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

    # remove NOOPs, which may include trivial permutes at this point
    OptimizationTranslations.apply_method_to_all_ops(REMOVE_NOOP, graph, fail_if_no_method=False)


# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------
def register(optimization_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param optimization_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(optimization_translation(), optimization_translation().op_type)
    return optimization_translation


@register
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.rank() == 4:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.rank() == 3:
	    buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.TBF_TO_BTF)
            buf.axis_format = AxisTracker.AxisFormat.BTF
            node.op.shape = buf.shape


@register
class OptimizeArgMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keepdims:
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                    AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
            axis_map = [0, 3, 1, 2]
            node.op.axis = axis_map[node.op.axis]


@register
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.image_to_spatial_first_order(node, graph)
        elif input_buf.rank() == 2 or input_buf.rank() == 3:
            if input_buf.rank() == 3:
                # add custom permute for 3D use-case. This input use-case is added for batchnorm-1D
                AxisTracker.enforce_input_format(graph, node.input_names[0],
                                                 AxisTracker.AxisFormat.NONTRIVIAL, [0, 2, 1])
            output_buf = graph.get_output_buffers(node)[0]
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank()))

    @staticmethod
    def squash_batchnorm(graph):
        def validate_input_rank(nodes_tuple):
            bn_node_ = nodes_tuple[1]
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            return bn_node_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY and bn_input_buffer_.rank() == 4

        sequence = [
                    ("convolution", (), ()),
                    ("batchnorm", (), ())

                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_input_rank)

        for node_tuple in matched_node_list:
            # sanity check
            log_assert(len(node_tuple) == len(sequence),
                       "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                       len(node_tuple), len(sequence))

            conv_node = node_tuple[0]
            bn_node = node_tuple[1]
            bn_input_buffer = graph.get_input_buffers(bn_node)[0]

            if bn_input_buffer.axis_format == AxisTracker.AxisFormat.NCS:
                # The Conv weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for BN weight broadcasting and then revert
                weights = numpy.transpose(conv_node.op.weights, (2, 3, 1, 0))
                weights = (weights * bn_node.op.weights)
                weights = numpy.transpose(weights, (3, 2, 0, 1))
            else:
                weights = (conv_node.op.weights * bn_node.op.weights)
            conv_node.op.weights = weights
            conv_node.op.bias = conv_node.op.bias * bn_node.op.weights + bn_node.op.bias
            graph.squash(bn_node, bn_input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                       conv_node.op.type,
                                                                                       conv_node.op.name))


@register
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def axes_to_snpe_order(self, node, graph):
        log_debug(code_to_message.get_debugging_message("DEBUG_AXES_TO_SNPE_ORDER_ENTRY")(node.op.name))
        super(OptimizeChannelShuffleTranslation, self).axes_to_spatial_first_order(node, graph)
        for buf in graph.get_input_buffers(node):
            log_debug("input {} {} {}", buf.name, buf.axis_format, buf.shape)
        for buf in graph.get_output_buffers(node):
            log_debug("output {} {} {}", buf.name, buf.axis_format, buf.shape)


@register
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeConvolutionTranslation, self).axes_to_spatial_first_order(node, graph)
        # if this method is called, current weight order for is NCHW but we want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 1, 0))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = [0, 3, 1, 2]
            node.op.axis = axis_map[node.op.axis]

    @staticmethod
    def fold_concats(graph):
        def validate_concat_axis(nodes_tuple):
            concat_node_ = nodes_tuple[0]
            concat_node_input_bufs_ = graph.get_input_buffers(concat_node_)
            for buf_ in concat_node_input_bufs_:
                if buf_.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_node_ = buf_.producer
                    # only fold concats with same axis
                    if prev_concat_node_.op.axis != concat_node_.op.axis:
                        log_debug2("Found concat node({}) with a concat input, but axis does not match for input ({}), "
                                   "{} != {} ", concat_node_.op.name, prev_concat_node_.op.name,
                                   prev_concat_node_.op.axis, concat_node_.op.axis)
                        return False

            return True

        sequence = [
                    ("concatenation",
                     ("FLEXIBLE_NUM_BUFS", [("concatenation", "ANY")]),
                     ()
                    )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_concat_axis)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_node_input_bufs = graph.get_input_buffers(concat_node)

            for buf in concat_node_input_bufs:
                if buf.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_buf = buf  # for readability
                    prev_concat_node = prev_concat_buf.producer

                    # remove prev concat as input from current concat and replace with prev concat's input names
                    prev_concat_inputs = prev_concat_node.input_names
                    idx = concat_node.input_names.index(prev_concat_buf.name)
                    concat_node.input_names.remove(prev_concat_buf.name)
                    concat_node.input_names[idx:idx] = prev_concat_inputs  # extend the inputs in the same index as prev concat

                    prev_concat_buf.consumers.remove(concat_node)

                    # we can prune the prev concat node if the current concat was the only consumer.
                    if len(prev_concat_buf.consumers) == 0:
                        graph.prune(prev_concat_node)

                    # remove prev concat as consumer for prev concat's input bufs and replace with current concat
                    for input_name in prev_concat_inputs:
                        input_buf = graph.get_buffer(input_name)
                        input_buf.consumers.add(concat_node)

                    log_debug2(code_to_message.get_debugging_message("DEBUG_CONCAT_FOLD")(prev_concat_node.op.name,
                                                                                          concat_node.op.name))


@register
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])


        # Permute the constant data if necessary
        if output_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC))
        elif output_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.TBF_TO_BTF))

        AxisTracker.eltwise_to_spatial_first_order(node, graph)

    @staticmethod
    def remove_noop(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)


@register
class OptimizeCropTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizeCrossCorrelationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CrossCorrelationOp.TRANSLATION_KEY


@register
class OptimizeDeconvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeDeconvolutionTranslation, self).axes_to_spatial_first_order(node, graph)

        # weights are in CNHW, want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 0, 1))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register
class OptimizeElementwiseMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY


@register
class OptimizeElementwiseProductTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseProductOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(graph):
        def validate_node(nodes_tuple):
            prod_node = nodes_tuple[0]
            if hasattr(prod_node.op, 'weights'):
                input_buffer_ = graph.get_input_buffers(prod_node)[0]
                prev_ = input_buffer_.producer
                log_assert(prev_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY,
                           code_to_message.get_error_message("ERROR_MUL_SCALE_PREV_NOT_BATCHNORM")(prev_.op.name,
                                                                                                   prev_.op.type))
                return True
            return False

        sequence = [
                    ("elementwise_product", (), ())
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            node = node_tuple[0]
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            weights = node.op.weights
            prev.op.weights *= weights
            prev.op.bias *= weights
            graph.squash(node, input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEPRODUCT_SQUASH")(node.op.name,
                                                                                                prev.op.name,
                                                                                                prev.op.type))


@register
class OptimizeElementwiseSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSumOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(graph):
        def validate_node(nodes_tuple):
            sum_node = nodes_tuple[0]
            if hasattr(sum_node.op, 'bias'):
                input_buffer_ = graph.get_input_buffers(sum_node)[0]
                prev_ = input_buffer_.producer
                log_assert(hasattr(prev_.op, 'bias'),
                           code_to_message.get_error_message("ERROR_ADD_BIAS_PREV_NO_BIAS")(sum_node.op.name,
                                                                                            prev_.op.name,
                                                                                            prev_.op.type))
                return True
            return False

        sequence = [
                    ("elementwise_sum", (), ())
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            node = node_tuple[0]
            input_buffer = graph.get_input_buffers(node)[0]
            prev = input_buffer.producer
            prev.op.bias += node.op.bias
            graph.squash(node, input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_ELEMENTWISESUM_SQUASH")(node.op.name,
                                                                                                prev.op.name,
                                                                                                prev.op.type))


@register
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_to_spatial_first_order(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.enforce_input_format(graph, input_buf.name, AxisTracker.AxisFormat.NSC,
                                             AxisTracker.AxisFormat.NCS_TO_NSC)
            # weights expect NCHW order, need to permute
            input_buf = graph.get_input_buffers(node)[0]
            batch, height, width, depth = input_buf.shape
            weights = node.op.weights_list[0]

            # TODO: this optimization was added based on onnx framework. Verify(Modify) if
            #       change is needed for other frameworks
            # ONNX defines FC as W^Tx + b,
            # so the weights have shape (batch, input_size, output_size)
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            log_assert(input_size == depth * height * width,
                       code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                      input_size,
                                                                                      (depth, height, width)))
            weights.shape = (depth, height, width, output_size)
            weights = numpy.transpose(weights, (3, 1, 2, 0))
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)
            weights.shape = (output_size, input_size)
            node.op.weights_list[0] = weights
        elif input_buf.rank() == 2:
            # again, need to transpose weights for spatial_first order
            weights = node.op.weights_list[0]
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
            node.op.weights_list[0] = weights

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.FEATURE

    @staticmethod
    def squash_batchnorm(graph):
        sequence = [
            ("fully_connected", (), ()),
            ("batchnorm", (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            # sanity check
            log_assert(len(node_tuple) == len(sequence),
                       "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                       len(node_tuple), len(sequence))

            fc_node = node_tuple[0]
            bn_node = node_tuple[1]
            bn_input_buffer = graph.get_input_buffers(bn_node)[0]

            weights_list = []
            if bn_input_buffer.axis_format == AxisTracker.AxisFormat.NCS:
                # The FC weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for BN weight broadcasting and then revert
                for weights in fc_node.op.weights_list:
                    weights = numpy.transpose(weights, (2, 3, 1, 0))
                    weights = (weights * bn_node.op.weights)
                    weights_list.append(numpy.transpose(weights, (3, 2, 0, 1)))
            else:
                weights_list = [(weights * bn_node.op.weights) for weights in fc_node.op.weights_list]

            fc_node.op.weights = weights_list
            fc_node.op.bias = fc_node.op.bias * bn_node.op.weights + bn_node.op.bias
            graph.squash(bn_node, bn_input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                       fc_node.op.type,
                                                                                       fc_node.op.name))


@register
class OptimizeGatherTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis and if needed permute it for NSC
        # In addition, output buffer axis tracking stays the same as input so long
        # as the rank of indices == 1. Otherwise it's non trivial as the rank will change
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        output_buf = graph.get_output_buffers(node)[0]
        if node.op.axis < 0:
           node.op.axis = node.op.axis+input_buf.rank()
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = [0, 3, 1, 2]
            node.op.axis = axis_map[node.op.axis]
            if indices_buf.rank() > 1:
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                    AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
        else:
           if indices_buf.rank() > 1:
               output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
           else:
               output_buf.axis_format = input_buf.axis_format

@register
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY


@register
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeLstmTranslation, self).axes_to_spatial_first_order(node, graph)

        # weights are expected to be NxK, we want KxN
        node.op['gate_weights'] = numpy.ascontiguousarray(node.op.gate_weights.transpose(), dtype=numpy.float32)
        node.op['recurrent_weights'] = numpy.ascontiguousarray(node.op.recurrent_weights.transpose(),
                                                               dtype=numpy.float32)


@register
class OptimizeMaxYTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MaxYOp.TRANSLATION_KEY


@register
class OptimizeNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NeuronOp.TRANSLATION_KEY


@register
class OptimizeNoopTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Noop.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    @staticmethod
    def remove_noop(node, graph):
        graph.squash(node, node.input_names[0])


@register
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PoolOp.TRANSLATION_KEY


@register
class OptimizePermuteTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PermuteOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # special case: transforming to NSC, will become noop
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                return
            else:
                # going to nontrivial
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                    AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.BTF
            else:
                AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.TBF,
                                                    AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
                output_buf.axis_format = AxisTracker. AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            if len(node.op.order) == 4:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif len(node.op.order) > 4:
                raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_TOO_MANY_DIMENSIONS")(node.op.order))
            else:
                # nothing to be done
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.order == list(range(len(node.op.order))):
            # this permute is trivial, remove it
            graph.squash(node, input_buffer.name)


@register
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY


@register
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY


@register
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)

    @staticmethod
    def product(nums):
        if len(nums) == 0:
            return 1
        else:
            return reduce(mul, nums)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.NCS,
                                                AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            AxisTracker.inject_implicit_permute(graph, input_name, AxisTracker.AxisFormat.TBF,
                                                AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            pass
        elif input_buf.axis_format == AxisTracker.AxisFormat.FEATURE:
            pass
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        if output_buf.rank() > 4:
            log_assert(self.product(output_buf.shape[:-4]) == 1,
                       code_to_message.get_error_message("ERROR_RESHAPE_BATCH_UNSUPPORTED"))
            output_buf.shape = output_buf.shape[-4:]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    @staticmethod
    def match_channelshuffle(graph):
        def is_valid_channelshuffle(nodes_tuple):
            def check_for_valid_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[3] and
                        reshape_1_input_shape[3] == reshape_1_output_shape[4])

            def check_for_valid_permute(node):
                # Assuming the input shape is N[GC']HW
                return node.op.type == op_adapter.PermuteOp.TRANSLATION_KEY and node.op.order == [0, 2, 1, 3, 4]

            def check_for_valid_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                        reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                        reshape_2_input_shape[3] == reshape_2_output_shape[2] and
                        reshape_2_input_shape[4] == reshape_2_output_shape[3])

            first_, second_, third_ = nodes_tuple
            input_shape_ = graph.get_input_buffers(first_)[0].shape
            output_shape_ = graph.get_output_buffers(third_)[0].shape

            return ((output_shape_ == input_shape_) and
                    check_for_valid_reshape_1(first_) and
                    check_for_valid_permute(second_) and
                    check_for_valid_reshape_2(third_))

        sequence = [
                    ("reshape",
                        (),
                        ("MATCH_NUM_BUFS", [("permute", "ALL")])
                    ),
                    ("permute",
                        (),
                        ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                    ),
                    ("reshape",
                        (),
                        ()
                    )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_channelshuffle)

        for node_tuple in matched_node_list:

                # ChannelShuffle Op found,
                # Squash Permute and 2nd Reshape Op and
                # Replace 1st ReshapeOp with ShuffleOp
                first, second, third = node_tuple
                third_input_buffer = graph.get_input_buffers(third)[0]
                graph.squash(third, third_input_buffer.name)

                second_input_buffer = graph.get_input_buffers(second)[0]
                graph.squash(second, second_input_buffer.name)

                output_shape = first.op.output_shape
                # Assuming the shape is N[GC']HW
                groups = output_shape[1]
                shuffle_op = op_adapter.ChannelShuffleOp(None, groups=groups)
                shuffle_op.name = graph.naming_policy.get_op_name(shuffle_op)
                graph.replace(first.op, shuffle_op)
                log_debug2(code_to_message.get_debugging_message("DEBUG_CHANNEL_SHUFFLE_REPLACE")(first.op.name,
                                                                                                  second.op.name,
                                                                                                  third.op.name,
                                                                                                  shuffle_op.name))


@register
class OptimizeRNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RNormOp.TRANSLATION_KEY


@register
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY


@register
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        log_assert(output_buf.shape[0] == 1, code_to_message.get_error_message("ERROR_MAX_ROI_POOL_BATCH_UNSUPPORTED"))
        output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC


@register
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        node.op.output_shape = AxisTracker.permute_shape(node.op.output_shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        AxisTracker.image_to_spatial_first_order(node, graph)


@register
class OptimizeRnnTransformationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.time_series_to_spatial_first_order(node, graph)

@register
class OptimizeSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        AxisTracker.feature_to_spatial_first_order(node, graph)


@register
class OptimizeStaticTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StaticOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        pass

    def remove_noop(self, node, graph):
        graph.prune(node)


@register
class OptimizeSubtractMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY


@register
class OptimizeUpsampleIndexBaseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY


@register
class OptimizeUpsampleSparseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleSparseOp.TRANSLATION_KEY
