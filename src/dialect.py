from xdsl.irdl import *
from xdsl.dialects.builtin import *
from dataclasses import dataclass


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "dtl.index"


@irdl_attr_definition
class TensorType(ParametrizedAttribute):
    name = "dtl.tensor"

    num_dim = ParameterDef(IntAttr)


@irdl_attr_definition
class ScalarType(ParametrizedAttribute):
    name = "dtl.scalar"


@irdl_op_definition
class LambdaOp(Operation):
    name = "dtl.lambda"

    inputs = VarOperandDef(TensorType)
    return_type = AttributeDef(TensorType)
    body = SingleBlockRegionDef()


@irdl_op_definition
class ReturnOp(Operation):
    name = "dtl.return"

    op = OperandDef(TensorType)


@irdl_op_definition
class IndexOp(Operation):
    name = "dtl.index"

    tensor = OperandDef(TensorType)
    indices = VarOperandDef(IndexType)
    res = ResultDef(ScalarType)


@irdl_op_definition
class DeIndexOp(Operation):
    name = "dtl.deindex"

    body = SingleBlockRegionDef()
    res = ResultDef(TensorType)


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
        self.ctx.register_op(ReturnOp)
        self.ctx.register_op(IndexOp)
        self.ctx.register_op(DeIndexOp)
        self.ctx.register_op(DeIndexYieldOp)
