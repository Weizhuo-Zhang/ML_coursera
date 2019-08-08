# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *

# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class OnnxChannelShuffleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('groups', 'i'))
        return op_adapter.ChannelShuffleOp(src_op.name, groups=params.groups)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ChannelShuffleOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxChannelShuffleTranslation(),
                                      onnx_type('Channel_Shuffle'),
                                      op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('max', 'f'),
                                    ('min', 'f'))
        return op_adapter.NeuronOp(src_op.name,
                                   "NEURON_RELU_MIN_MAX",
                                   min_clamp=params.min,
                                   max_clamp=params.max)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.NeuronOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxClipTranslation(), onnx_type('Clip'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis', 'i'))

        # static concatenation used for reshaping shape tensors
        if graph.weights.has_all(src_op.input):
            data = [graph.weights.fetch(input_name) for input_name in src_op.input]
            concat_data = numpy.concatenate(data, params.axis)
            graph.weights.insert(str(src_op.output[0]), concat_data)
            return op_adapter.StaticOp(src_op.name)

        return op_adapter.ConcatOp(src_op.name, params.axis)

    def infer_output_shapes(self, op, input_shapes):
        # Add batch dim
        axis = op.axis
        output_shape = input_shapes[0][:]
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        return [output_shape]

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return [str(src_op.output[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ConcatOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxConcatTranslation(),
                                      onnx_type('Concat'),
                                      op_adapter.ConcatOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Constant
# ------------------------------------------------------------------------------
class OnnxConstantTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('value', 't'))
        graph.weights.insert(src_op.output[0], params.value)
        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(src_op.output[0], params.value)

    def infer_output_shapes(self, op, input_shapes):
        return [op.tensor.shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ConstantOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxConstantTranslation(),
                                      onnx_type('Constant'),
                                      op_adapter.ConstantOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('axis', 'i', 1))
        axis = params.axis

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [ product(pre_axes), product(post_axes) ]

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ReshapeOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxFlattenTranslation(), onnx_type('Flatten'))

#------------------------------------------------------------------------------
#   Gather
#------------------------------------------------------------------------------
class OnnxGatherTranslation(OnnxTranslationBase):
    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        if len(ops) > 1:
            graph.add(ops[0], [], input_names[1])
        graph.add(ops[-1], input_names, output_names)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('axis','i',0))
        ops = []
        indices_name = str(src_op.input[1])
        # If the input is stored as weights we need to create a const node
        if not graph.has_buffer(indices_name):
            indices = graph.weights.fetch(indices_name, prunable=False)
            ops.append(op_adapter.ConstantOp(indices_name, indices, quantizable=False))
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
               indices_op.quantizable = False
        ops.append(op_adapter.GatherOp(src_op.name, axis=params.axis))
        return ops

    def infer_output_shapes(self, op, input_shapes):
        output_shape = input_shapes[0][:op.axis]+list(input_shapes[1])+input_shapes[0][op.axis+1:]
        return [output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.GatherOp.TRANSLATION_KEY]

OnnxTranslations.register_translation(OnnxGatherTranslation(),
                                      onnx_type('Gather'),
                                      op_adapter.GatherOp.TRANSLATION_KEY)

# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class OnnxPadTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        pads = []

        # Extract op version specific padding information
        version = OpVersionInfo.onnx_op_ver(src_op, self.get_supported_version())
        if version == 1:
            pads = extract_attributes(src_op, ('paddings', 'li')).pads
        else:
            pads = extract_attributes(src_op, ('pads', 'li')).pads

        params = extract_attributes(src_op,
                                    ('mode', 's'),
                                    ('value', 'f', 0))

        supported_modes = {'constant': "PADDING_CONSTANT",
                           'reflect': "PADDING_REFLECT"}
        if params.mode not in supported_modes:
            log_error(code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(params.mode))

        # Pads/paddings need to be translated from r1_begin, r2_begin...r1_end, r2_end, ...
        # to pairs (r1_begin, r1_end), (r2_begin, r2_end)...
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)
        log_assert(rank == len(pads)/2,
                   "Rank of input tensor: %d must equal (# pads/2): %d",
                   rank,
                   len(pads)/2)

        pad_pairs = []
        for index in range(rank):
            pad_pairs.append([pads[index], pads[index+rank]])
        return op_adapter.PadOp(src_op.name,
                                mode=supported_modes[params.mode],
                                pads=pad_pairs,
                                constant_value=params.value)

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        output_shape = []

        for i in range(0, len(input_shape)):
            output_shape.append(input_shape[i] + op.pads[i][0] + op.pads[i][1])

        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, output_shape))
        return [output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.PadOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxPadTranslation(),
                                      onnx_type('Pad'),
                                      op_adapter.PadOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        # There are two main versions of ONNX Reshape
        #    1. The old reshape, where shape is provided as an attribute
        #    2. The new reshape, where the shape is provided as a second input
        #
        # SNPE and the converter support two versions of Reshape:
        #    1. Dynamic reshaping with a statically provided output shape
        #    2. Static reshaping, performed at conversion time
        #
        # SNPE can't support the 2nd ONNX Reshape expclicitly, however we can
        # calculate the shape ahead of time and statically set in in the SNPE layer.
        # This will prevent the network from being resizable. In addition, if a
        # 'Shape' layer provided the shape it will have been saved as static,
        # eg weight data, in the converter and all ops operating on that data will
        # become static ops and will be pruned during the final conversion.
        shape = []
        if len(src_op.input) > 1:
            shape_input = str(src_op.input[1])
            if graph.weights.has(shape_input):
                shape = graph.weights.fetch(shape_input).astype(numpy.int64).tolist()
            else:
                shape = graph.get_buffer(str(src_op.input[1])).shape.tolist()
        else:
            params = extract_attributes(src_op, ('shape', 'li'))
            shape = params.shape

        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)
        else:
            # dynamic reshape of activations
            input_buf = graph.get_buffer(input_name)
            input_shape = input_buf.shape

            remainder_size = product(input_shape)
            remainder_index = -1
            output_shape = []
            for i, s in enumerate(shape):
                if s == -1:
                    remainder_index = i
                    output_shape.append(0)
                elif s == 0:
                    remainder_size /= input_shape[i]
                    output_shape.append(input_shape[i])
                else:
                    remainder_size /= s
                    output_shape.append(s)
            if remainder_index >= 0:
                output_shape[remainder_index] = remainder_size

            return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def get_supported_version(self):

        return OP_VERSION_SUPPORTED[op_adapter.ReshapeOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxReshapeTranslation(),
                                      onnx_type('Reshape'),
                                      op_adapter.ReshapeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Shape
# ------------------------------------------------------------------------------
class OnnxShapeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_STATIC_SHAPE")(src_op.name))
        shape = graph.get_buffer(str(src_op.input[0])).shape
        output_name = str(src_op.output[0])
        graph.weights.insert(output_name, numpy.asarray(shape, dtype=numpy.int64))
        return op_adapter.StaticOp(src_op.name)

    def extract_input_names(self, src_op, graph):
            return []

    def extract_output_names(self, src_op, graph):
            return []

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED['shape']


OnnxTranslations.register_translation(OnnxShapeTranslation(),
                                      onnx_type('Shape'))


# ------------------------------------------------------------------------------
#   Slice, Crop
# ------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op,
                                    ('axes', 'li', []),
                                    ('ends', 'li'),
                                    ('starts', 'li'))
        # If axes are not provided axes is set from # of 'starts'
        if not params.axes:
            params.axes = list(range(len(params.starts)))

        log_assert(len(params.starts) == len(params.axes),
                   "Node %s: expected same number of starts as axes",
                   src_op.name)
        log_assert(len(params.ends) == len(params.axes),
                   "Node %s: expected same number of ends as axes",
                   src_op.name)

        def get_indicies(start, end, dim):
            # Negative values mean wrap around, like in python
            if start < 0:
                start = int(start % dim)
            if end < 0:
                end = int(end % dim)
            # higher than the size, however, means stop at the end.
            start = min(start, dim)
            end = min(end, dim)
            return start, end

        # Static slicing used for shape tensors
        if graph.weights.has(input_name):
            data = graph.weights.fetch(input_name)
            for i in range(len(params.axes)):
                start, end = get_indicies(params.starts[i], params.ends[i], data.shape[params.axes[i]])
                data = data.take(indices=list(range(start, end)), axis=params.axes[i])
            output_name = str(src_op.output[0])
            graph.weights.insert(output_name, data)
            return op_adapter.StaticOp(src_op.name)

        # canonicalize the axes
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = input_buf.rank()
        offsets = [0]*rank
        output_shape = list(input_buf.shape[:])
        for i, axis in enumerate(params.axes):
            start = params.starts[i]
            end = params.ends[i]
            dim = input_buf.shape[axis]
            start, end = get_indicies(start, end, dim)
            offsets[axis] = start
            output_shape[axis] = end-start

        return op_adapter.CropOp(src_op.name, offsets, output_shape)

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.output))

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.CropOp.TRANSLATION_KEY]


# Onnx Crop should go here as well, but the documentation is really
# ambiguous so we won't add it until we see an example.
OnnxTranslations.register_translation(OnnxSliceTranslation(),
                                      onnx_type('Slice'),
                                      op_adapter.CropOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis', 'i'),
                                    ('split', 'li', []))
        input_buf = graph.get_buffer(str(src_op.input[0]))
        if not params.split:
            params.split = [input_buf.shape[params.axis]/len(src_op.output)]

        slice_points = []
        next_slice_point = 0
        for split in params.split[1:]:
            next_slice_point += split
            slice_points.append(next_slice_point)
        return op_adapter.SliceOp(src_op.name,
                                  axis=params.axis,
                                  slice_points=slice_points)

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.SliceOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxSplitTranslation(),
                                      onnx_type('Split'),
                                      op_adapter.SliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class OnnxSqueezeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_name = str(src_op.input[0])
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape
        default_axes = [i for i, s in enumerate(input_shape) if s == 1]
        params = extract_attributes(src_op, ('axes', 'li', default_axes))

        if not all(x < len(input_shape) for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIM_GREATER_THAN_RANK")(params.axes,
                                                                                                      len(input_shape)))

        if not all((input_shape[x] == 1) for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIMS_EQUAL_ONE")(params.axes,
                                                                                               input_shape))

        output_shape = [s for i, s in enumerate(input_shape) if i not in params.axes]

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED['squeeze']


OnnxTranslations.register_translation(OnnxSqueezeTranslation(), onnx_type('Squeeze'))


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('perm', 'li'))
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            w = numpy.transpose(w, params.perm)
            graph.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, w.shape))

            return op_adapter.StaticOp(src_op.name)

        return op_adapter.PermuteOp(src_op.name, params.perm)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        # return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def infer_output_shapes(self, op, input_shapes):
        output_shape = [input_shapes[0][i] for i in op.order]
        return [output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.PermuteOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxTransposeTranslation(),
                                      onnx_type('Transpose'),
                                      op_adapter.PermuteOp.TRANSLATION_KEY)


# -----------------------------------------------------------------------------
#   Unsqueeze
# ------------------------------------------------------------------------------
class OnnxUnsqueezeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # default_axes = [i for i, s in enumerate(input_shape) if s == 1]
        params = extract_attributes(src_op, ('axes', 'li'))

        input_name = str(src_op.input[0])
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape
        if not all(x >= 0 for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_NEGATIVE_DIMS")(params.axes))

        new_rank = len(input_shape)+len(params.axes)
        if not all(x < new_rank for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DIMS_GREATER_THAN_RANK")(params.axes,
                                                                                                         new_rank))

        if len(set(params.axes)) != len(params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DUPLICATE_DIMS")(params.axes))

        params.axes.sort()
        output_shape = input_shape
        for i in params.axes:
            output_shape.insert(i, 1)

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # Otherwise this is a dynamic unsqueeze so add the unsqueeze/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED['unsqueeze']


OnnxTranslations.register_translation(OnnxUnsqueezeTranslation(), onnx_type('Unsqueeze'))


# ------------------------------------------------------------------------------
#   Upsample
# ------------------------------------------------------------------------------
class OnnxUpsampleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('mode', 's', 'nearest'),
                                    ('scales', 'lf'))
        input_buf = graph.get_buffer(str(src_op.input[0]))
        if input_buf.rank() != 4:
            raise ValueError(code_to_message.get_error_message("ERROR_UPSAMPLE_INPUT_DIMS")(input_buf.shape))
        scale_height = params.scales[2]
        scale_width = params.scales[3]
        supported_modes = {'nearest': "RESIZE_NEAREST_NEIGHBOR",
                           'bilinear': "RESIZE_BILINEAR"}

        if params.mode not in supported_modes:
            raise ValueError(code_to_message.get_error_message("ERROR_UPSAMPLE_UNSUPPORTED_MODE")(params.mode))
        mode = supported_modes[params.mode]

        # Generate output shape using output_dims = floor(input_dims * scale).
        input_shape = input_buf.shape
        input_height = input_shape[2]
        input_width = input_shape[3]
        output_height = int(input_height * scale_height)
        output_width = int(input_width * scale_width)
        output_shape = [input_shape[0], input_shape[1], output_height, output_width]
        return op_adapter.ResizeOp(src_op.name,
                                   output_shape,
                                   resize_mode=mode,
                                   scale_height=scale_height,
                                   scale_width=scale_width)

    def infer_output_shapes(self, op, input_shapes):
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, op.output_shape))
        return [op.output_shape]

    def get_supported_version(self):
        return OP_VERSION_SUPPORTED[op_adapter.ResizeOp.TRANSLATION_KEY]


OnnxTranslations.register_translation(OnnxUpsampleTranslation(),
                                      onnx_type('Upsample'),
                                      op_adapter.ResizeOp.TRANSLATION_KEY)
