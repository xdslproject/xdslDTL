from dialect import *

from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, op_type_rewrite_pattern, PatternRewriter, PatternRewriteWalker
from xdsl.ir import MLContext, Operation, SSAValue, Region, Block, Attribute
from dataclasses import dataclass

import xdsl.dialects.memref as memref
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.builtin as builtin

tensor_shape: dict[str, int] = {}
tensor_shape["P"] = 3
tensor_shape["Q"] = 4

tensor_type = builtin.f32

output_buf = 1

@dataclass
class IndexRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, index_op: IndexOp, rewriter: PatternRewriter):

        load_op = memref.Load.get(index_op.tensor, index_op.indices)
        store_op = memref.Store.get(load_op, index_op.tensor, index_op.indices)
        id_op = arith.Constant.from_int_constant(3, 32)
        rewriter.replace_op(index_op, [load_op, store_op, id_op])


@dataclass
class DeIndexOpRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, deindex_op: DeIndexOp,
                          rewriter: PatternRewriter):
        new_ops = []
        outer_len = tensor_shape[
            deindex_op.body.blocks[0].args[0].typ.parameters[0].data]
        inner_len = tensor_shape[
            deindex_op.body.blocks[0].args[1].typ.parameters[0].data]
        output = memref.Alloca.get(tensor_type, 4, [outer_len, inner_len])

        output_buf = output
        new_ops.append(output)

        outer_ind_op = arith.Constant.from_int_constant(0, 32)
        new_ops.append(outer_ind_op)
        outer_len_op = arith.Constant.from_int_constant(outer_len, 32)
        new_ops.append(outer_len_op)
        inner_ind_op = arith.Constant.from_int_constant(0, 32)
        new_ops.append(inner_ind_op)
        inner_len_op = arith.Constant.from_int_constant(inner_len, 32)
        new_ops.append(inner_len_op)

        one_op = arith.Constant.from_int_constant(1, 32)
        new_ops.append(one_op)

        outer_comp_op = arith.Cmpi.get(outer_ind_op, outer_len_op, 6)
        outer_inc_op = arith.Addi.get(outer_ind_op, one_op)
        outer_comp_ops = [outer_comp_op]

        inner_comp_op = arith.Cmpi.get(inner_ind_op, inner_len_op, 6)
        inner_inc_op = arith.Addi.get(inner_ind_op, one_op)
        inner_comp_ops = [inner_comp_op]

        inner_while = scf.While.build(
            operands=[[]],
            result_types=[[
                memref.MemRefType.from_type_and_list(IntAttr.from_int(3),
                                                     [outer_len, inner_len])
            ]],
            regions=[
                Region.from_operation_list(inner_comp_ops),
                Region.from_operation_list([])
            ])

        block = deindex_op.body.detach_block(deindex_op.body.blocks[0])
        inner_while.after_region.insert_block(block, 0)
        inner_while.after_region.blocks[0].add_op(inner_inc_op)

        outer_while = scf.While.build(
            operands=[[]],
            result_types=[[
                memref.MemRefType.from_type_and_list(IntAttr.from_int(3),
                                                     [outer_len, inner_len])
            ]],
            regions=[
                Region.from_operation_list(outer_comp_ops),
                Region.from_operation_list([inner_while])
            ])
        outer_while.after_region.blocks[0].add_op(outer_inc_op)
        new_ops.append(outer_while)

        rewriter.replace_op(deindex_op, new_ops)


# @dataclass
# class LambdaRewriter():
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, lambda_op: LambdaOp,
#                           rewriter: PatternRewriter):
#         outer_len = tensor_shape[
#             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[0].data]
#         inner_len = tensor_shape[
#             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[1].data]
#         type_ = memref.MemRefType.from_type_and_list(IntAttr.from_int(2),
#                                                      [outer_len, inner_len])
#
#         lambda_op.body.blocks[0].args[0].typ = type_


def transform_dtl(ctx: MLContext, op: Operation):
    applier = PatternRewriteWalker(GreedyRewritePatternApplier(
        [DeIndexOpRewriter(),
         # LambdaRewriter(),
         IndexRewriter()]),
                                   walk_regions_first=False)

    applier.rewrite_module(op)
