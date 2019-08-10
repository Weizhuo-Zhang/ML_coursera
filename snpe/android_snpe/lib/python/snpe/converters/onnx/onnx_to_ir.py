# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import argparse
import traceback
import sys

try:
    import onnx
    parse_model = onnx.load
except ImportError:
    def parse_model(model_path):
        raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(sys.path)))

from snpe.converters.common.utils import snpe_converter_utils
from snpe.converters.common.converter_ir import op_graph, op_graph_optimizations, op_policies, translation
from .util import *
from . import onnx_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverter(object):
    def __init__(self):
        self.translations = onnx_translations.OnnxTranslations
        self.graph = op_graph.IROpGraph(naming_policy=OnnxNamePolicy(),
                                        shape_inference_policy=OnnxShapeInferencePolicy())
        self.input_model_path = ''
        self.output_model_path = ''
        self.converter_command = ''
        self.copyright_str = ''
        self.op_info = onnx_translations.OpVersionInfo()
        self.debug = False
        self.disable_batchnorm_folding = False
        self.set_options(self.parse_args())

    @staticmethod
    def parse_args():
        """
        Command Line Processing
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", "-m", help="Path to the source ONNX model.",
                            required=True)
        parser.add_argument("--output_path", "-d", help="Path where the converted Output model should be saved.")
        parser.add_argument('--copyright_file', type=str,
                            help='Path to copyright file. If provided, the content of the file will be added to the '
                                 'output model.')
        parser.add_argument('--encoding',
                            help='Set the image encoding for an input buffer. This should be specified in the format '
                                 '"--encoding <input name> <encoding>", where encoding is one of: "argb32", "rgba", "nv21",'
                                 ' "opaque", or "bgr". The default encoding for all inputs not so described is "bgr". '
                                 '"opaque" inputs will be interpreted as-is, and not subject to order transformations.',
                            nargs=2, action='append')
        parser.add_argument("--disable_batchnorm_folding", help="If not specified, converter will try to fold batchnorm "
                                                                "into previous convolution layer", action="store_true")

        parser.add_argument("--debug", type=int, nargs='?', const=0, default=-1, help="Run the converter in debug mode.")
        args = parser.parse_args()

        return args

    def set_options(self, args):
        setup_logging(args)
        self.input_model_path = args.model_path
        self.output_model_path = args.output_path
        self.debug = args.debug
        self.disable_batchnorm_folding = args.disable_batchnorm_folding
        self.converter_command = snpe_converter_utils.sanitize_args(args, args_to_ignore=['model_path', 'm',
                                                                                          'output_path', 'd'])
        self.copyright_str = snpe_converter_utils.get_string_from_txtfile(args.copyright_file)

    def convert(self):
        model = parse_model(self.input_model_path)
        self.op_info.set_global_op_ver(model)
        self.graph.weights = WeightProvider(model)
        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(onnx_type("input"),
                                                 onnx_translations.ADD_INPUT_OP, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = onnx_type(src_op.op_type)
            try:
                supported_version = self.translations.apply_method_to_op(src_type,
                                                                         onnx_translations.SUPPORTED_VERSION)
                self.op_info.validate_op_ver(src_op, supported_version)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

            try:
                self.translations.apply_method_to_op(src_type,
                                                     translation.ADD_OP,
                                                     src_op,
                                                     self.graph)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        return self.graph

    def ir_optimize(self, graph, **kwargs):
        try:
            # apply graph transformations
            op_graph_optimizations.apply_graph_optimizations(graph, self.disable_batchnorm_folding, **kwargs)
            return graph
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            log_error(str(e))
            sys.exit(-1)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        if op.name:
            return str(op.name)
        else:
            count = self.type_count.get(op.type, 0)
            self.type_count[op.type] = count+1
            return "%s_%d" % (op.type, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     translation.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
