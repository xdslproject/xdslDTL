from xdsl.irdl import *
from xdsl.dialects.builtin import *
from dataclasses import dataclass


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "dtl.index"
    
    dim: ParameterDef[StringAttr]


@irdl_attr_definition
class TensorType(ParametrizedAttribute):
    name = "dtl.tensor"

    dims: ParameterDef[ArrayAttr[StringAttr | IntAttr]]


@irdl_attr_definition
class ScalarType(ParametrizedAttribute):
    name = "dtl.scalar"


@irdl_op_definition
class LambdaOp(Operation):
    name = "dtl.lambda"

    inputs = VarOperandDef(TensorType)
    return_type = AttributeDef(TensorType)
    func_name = AttributeDef(StringAttr)
    body = SingleBlockRegionDef()
    
    def verify_(self):
        ret = self.body.ops[-1]
        if not isinstance(ret, LambdaYieldOp):
            raise Exception(
                f"{LambdaYieldOp.name} expected as last operation of a {LambdaOp.name} node"
            )
        if ret.op.typ != self.return_type:
            raise Exception(
                f"{LambdaOp.name} should have a {LambdaYieldOp.name} with the same return type"
            )

    def get_inputs(self):
        return self.body.blocks[0].args


@irdl_op_definition
class LambdaYieldOp(Operation):
    name = "dtl.return"

    op = OperandDef(TensorType)

    def verify_(self):
        if not isinstance(self.parent.parent.parent, LambdaOp):
            raise Exception(
                f"Parent of {LambdaYieldOp.name} should be a {LambdaOp.name}")


@irdl_op_definition
class IndexOp(Operation):
    name = "dtl.index"

    tensor = OperandDef(TensorType)
    indices = VarOperandDef(IndexType)
    res = ResultDef(ScalarType)

    def verify_(self):
        if len(self.indices) != len(self.tensor.typ.dims.data):
            raise Exception(
                f"An {IndexOp.name} should index a tensor with as many indices as its dimension"
            )
        for (idx, tensor_idx) in zip(self.indices, self.tensor.typ.dims.data):
            if idx.typ.dim.data != tensor_idx.data:
                raise Exception(
                    f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
                )


@irdl_op_definition
class DeIndexOp(Operation):
    name = "dtl.deindex"

    body = SingleBlockRegionDef()
    res = ResultDef(TensorType)

    def verify_(self):
        if len(self.body.blocks[0].args) != len(self.res.typ.dims.data):
            raise Exception(
                f"An {DeIndexOp.name} should return a tensor with as many dimensions as the index it produces"
            )
        for (idx, tensor_idx) in zip(self.body.blocks[0].args,
                                     self.res.typ.dims.data):
            if idx.typ.dim.data != tensor_idx.data:
                raise Exception(
                    f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
                )

        ret = self.body.ops[-1]
        if not isinstance(ret, DeIndexYieldOp):
            raise Exception(
                f"{DeIndexYieldOp.name} expected as last operation of a {DeIndexOp.name} node"
            )

    def get_ssa_indices(self):
        return self.body.blocks[0].args


@irdl_op_definition
class DeIndexYieldOp(Operation):
    name = "dtl.deindex_yield"

    op = OperandDef(ScalarType)



@irdl_op_definition
class SumOp(Operation):
    name = "dtl.sum"

    body = SingleBlockRegionDef()
    res = ResultDef(ScalarType)

    def verify_(self):
        if len(self.body.blocks[0].args) == 0:
            raise Exception(
                f"A {SumOp.name} should sum over at least one index"
            )
        for idx in self.body.blocks[0].args:
            if not isinstance(idx.typ, IndexType):
                raise Exception(f"A {SumOp.name} may only sum over an Indextype (args of enclosed Block)")
        # if len(self.body.blocks[0].args) != len(self.res.typ.dims.data):
        #     raise Exception(
        #         f"A {SumOp.name} should return a tensor with as many dimensions as the index it produces"
        #     )
        # for (idx, tensor_idx) in zip(self.body.blocks[0].args,
        #                              self.res.typ.dims.data):
        #     if idx.typ.dim.data != tensor_idx.data:
        #         raise Exception(
        #             f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
        #         )

        ret = self.body.ops[-1]
        if not isinstance(ret, SumYieldOp):
            raise Exception(
                f"{SumYieldOp.name} expected as last operation of a {SumOp.name} node"
            )

    def get_ssa_indices(self):
        return self.body.blocks[0].args


@irdl_op_definition
class SumYieldOp(Operation):
    name = "dtl.sum_yield"

    op = OperandDef(ScalarType)



@irdl_op_definition
class ScalarAddOp(Operation):
    name = "dtl.scalarAdd"

    lhs = OperandDef(ScalarType)
    rhs = OperandDef(ScalarType)
    res = ResultDef(ScalarType)
    

@irdl_op_definition
class ScalarMulOp(Operation):
    name = "dtl.scalarMul"

    lhs = OperandDef(ScalarType)
    rhs = OperandDef(ScalarType)
    res = ResultDef(ScalarType)


@dataclass
class DTL:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(IndexType)
        self.ctx.register_attr(TensorType)
        self.ctx.register_attr(ScalarType)

        self.ctx.register_op(LambdaOp)
        self.ctx.register_op(LambdaYieldOp)
        self.ctx.register_op(IndexOp)
        self.ctx.register_op(ScalarAddOp)
        self.ctx.register_op(ScalarMulOp)
        self.ctx.register_op(DeIndexOp)
        self.ctx.register_op(DeIndexYieldOp)
        self.ctx.register_op(SumOp)
        self.ctx.register_op(SumYieldOp)
