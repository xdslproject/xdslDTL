from xdsl.irdl import *
from xdsl.dialects.builtin import *
from dataclasses import dataclass


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "dtl.index"

    dim = ParameterDef(StringAttr)


@irdl_attr_definition
class TensorType(ParametrizedAttribute):
    name = "dtl.tensor"

    dims = ParameterDef(ArrayOfConstraint(AnyOf([StringAttr, IntAttr])))


@irdl_attr_definition
class ScalarType(ParametrizedAttribute):
    name = "dtl.scalar"


@irdl_op_definition
class LambdaOp(Operation):
    name = "dtl.lambda"

    inputs = VarOperandDef(TensorType)
    return_type = AttributeDef(TensorType)
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


@irdl_op_definition
class DeIndexYieldOp(Operation):
    name = "dtl.deindex_yield"

    op = OperandDef(ScalarType)


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
        self.ctx.register_op(DeIndexOp)
        self.ctx.register_op(DeIndexYieldOp)
