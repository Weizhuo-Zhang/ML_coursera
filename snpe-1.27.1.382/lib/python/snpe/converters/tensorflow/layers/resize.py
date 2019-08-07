#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class ResizeBilinearLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op, align_corners=False, mul_const = [0, 0]):
            super(ResizeBilinearLayerResolver.Descriptor, self).__init__('Resize', name, nodes)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = 0
            self.resize_op = resize_op
            self.mul_const = mul_const

        def is_input_tensor(self, op, tensor):
            if op == self.resize_op and tensor != self.resize_op.inputs[0]:
                return False
            return True

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeBilinear'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeBilinear'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
                input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
                input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
                consumed_nodes = match.consumed_nodes
                mul_const = [0, 0]

                if('mul_const' in match):
                    mul_const_op = match['mul_const']
                    mul_const = graph_helper.evaluate_tensor_output(mul_const_op.outputs[0])
                    if(len(mul_const) < 2):
                        mul_const = [0, 0]

                descriptors.append(
                    ResizeBilinearLayerResolver.Descriptor(str(resize_op.name),
                                                           consumed_nodes,
                                                           input_tensor_shape,
                                                           resize_op,
                                                           align_corners_bool,
                                                           mul_const))

        return descriptors


class ResizeNearestNeighborLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op, align_corners=False, mul_const = [0, 0]):
            super(ResizeNearestNeighborLayerResolver.Descriptor, self).__init__('ResizeNearestNeighbor', name, nodes)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = 1
            self.resize_op = resize_op
            self.mul_const = mul_const

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
                input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
                input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
                consumed_nodes = match.consumed_nodes
                mul_const = [0, 0]

                if('mul_const' in match):
                    mul_const_op = match['mul_const']
                    mul_const = graph_helper.evaluate_tensor_output(mul_const_op.outputs[0])
                    if(len(mul_const) < 2):
                        mul_const = [0, 0]

                descriptors.append(
                    ResizeNearestNeighborLayerResolver.Descriptor(str(resize_op.name),
                                                                  consumed_nodes,
                                                                  input_tensor_shape,
                                                                  resize_op,
                                                                  align_corners_bool,
                                                                  mul_const))

        return descriptors

class ResizeLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.resize_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return converter_context.model.add_scaling_layer(descriptor.output_names[0],
                                                         output_shape,
                                                         pad_value=0.0,
                                                         maintain_aspect_ratio=False,
                                                         resize_mode=descriptor.resize_mode,
                                                         scale_height=descriptor.mul_const[0],
                                                         scale_width=descriptor.mul_const[1],
                                                         input_name=input_name,
                                                         output_name=descriptor.output_names[0],
                                                         align_corners=descriptor.align_corners)

