from __future__ import annotations

import builtins
from xdsl.dialects import builtin, arith
from xdsl.ir import _PA, MLContext
from xdsl.irdl import *
from xdsl.dialects.builtin import *
from dataclasses import dataclass

from xdsl.utils.hints import isa

IndexT = TypeVar("IndexT", bound=Attribute)


@irdl_attr_definition
class IndexShapeStruct(Generic[IndexT], ParametrizedAttribute):
    name : str = "dtl.IndexShapeStruct"
    shape: ParameterDef[ArrayAttr[Attribute]]
    
    def verify(self):
        assert isa(self.shape, ArrayAttr[Attribute])
        
    def verify_generic(self, type:Type):
        for i in self.shape.data:
            assert isa(i, type), f"{self.name}:: type missmatch. Expecting {type.name if isclass(type) and issubclass(type, Attribute) else type}, found {i}"
        


@irdl_attr_definition
class IndexTupleStruct(Generic[IndexT], ParametrizedAttribute):
    name : str = "dtl.IndexStruct"
    children: ParameterDef[ArrayAttr[Attribute]]
    
    def verify(self):
        assert isa(self.children, ArrayAttr[Attribute])
        
    def verify_generic(self, type:Type):
        for child in self.children.data:
            child.verify_generic(type)



IndexStruct: TypeAlias = IndexTupleStruct | IndexShapeStruct

@irdl_attr_definition
class Index(ParametrizedAttribute):
    name : str = "dtl.index"
    id: ParameterDef[StringAttr]

@irdl_attr_definition
class KnownVectorSpace(ParametrizedAttribute):
    name : str = "dtl.KnownVectorSpace"
    dim: ParameterDef[IntAttr]
    
@irdl_attr_definition
class UnknownVectorSpace(ParametrizedAttribute):
    name : str = "dtl.UnknownVectorSpace"
    id: ParameterDef[StringAttr]
    
VectorSpace: TypeAlias = UnknownVectorSpace | KnownVectorSpace

@irdl_attr_definition
class IndexToVectorSpaceMapPair(ParametrizedAttribute):
    name : str = "dtl.indexToVectorSpaceMapPair"
    index: ParameterDef[Index]
    vector_space: ParameterDef[VectorSpace]
    
@irdl_attr_definition
class IndexToVectorSpaceMap(ParametrizedAttribute):
    name : str = "dtl.indexToVectorSpaceMap"
    mapping: ParameterDef[ArrayAttr[IndexToVectorSpaceMapPair]]

    def __init__(self, params:list[Attribute]) -> None:
        print(params)
        print("==========\nNEW IndexToVectorSpaceMap!\n=============")
        raise NotImplementedError
        super().__init__(params)
    
    def verify(self):
        index_names = [pair.index.id.data for pair in self.mapping.data]
        assert index_names == sorted(index_names), "IndexToVectorSpaceMap:: IndexToVectorSpaceMapPairs must be ordered by the id of the indices"
        assert len(index_names) == len(set(index_names)), "IndexToVectorSpaceMap:: Duplicate keys found"
        
    def indices(self):
        return [pair.index for pair in self.mapping.data]
    
    def vector_space_of(self, index: Index):
        l = [pair.vector_space for pair in self.mapping.data if pair.index == index]
        if len(l) == 0:
            raise KeyError("index not found in IndexToVectorSpaceMap")
        if len(l) != 1:
            raise KeyError("IndexToVectorSpaceMap has duplicates - Verification was not used?")
        return l[0]

# @irdl_attr_definition
# class TensorDimType(ParametrizedAttribute):
#     name : str = "dtl.tensorResultDim"
#     dims: ParameterDef[IntAttr]

TensorResultType: TypeAlias = IndexStruct[VectorSpace]

@irdl_attr_definition
class TensorExprType(ParametrizedAttribute):
    name : str = "dtl.tensorExprType"
    args: ParameterDef[IndexToVectorSpaceMap]
    result: ParameterDef[TensorResultType]
    
    def getIndices(self):
        return self.args.indices()
    
    def verify(self) -> None:
        self.result.verify_generic(VectorSpace)


@irdl_attr_definition
class NoneIndex(ParametrizedAttribute):
    name : str = "dtl.NoneIndex"

IndexingStruct: TypeAlias = IndexStruct[Index | NoneIndex]
DeIndexingStruct: TypeAlias = IndexStruct[Index | VectorSpace]

@irdl_op_definition
class IndexBindingOp(Operation):
    name : str = "dtl.bind"

    expr: Annotated[Operand, TensorExprType]
    indices_map: OpAttr[IndexToVectorSpaceMap]
    result: Annotated[OpResult, TensorExprType]
    def verify_(self):
        print(self.indices_map)
        for idx in self.indices_map.indices():
            if idx in self.expr.typ.args.indices():
                raise Exception(
                    f"An {IndexBindingOp.name} should can only bind indices that are not already bound in its subexpression. {idx} is already bound in {self.expr}"
                )

def matchTensorTupleStructures(typeResult: TensorResultType, indexStruct: IndexStruct, DeIndexing=False) -> bool:
    print("matchTensorTupleStructures")
    return False

@irdl_op_definition
class IndexOp(Operation):
    name : str = "dtl.index"

    expr: Annotated[Operand, TensorExprType]
    indices: OpAttr[IndexingStruct]
    result: Annotated[OpResult, TensorExprType]

    def verify_(self):
        self.indices.verify_generic(Index | NoneIndex)
        if not matchTensorTupleStructures(self.expr.typ.result, self.indices):
            raise Exception(f"IndexOp indicies do not match type of given expression")
        raise NotImplementedError
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
    name : str = "dtl.deindex"
    
    expr: Annotated[Operand, TensorExprType]
    indices: Annotated[Operand, DeIndexingStruct]
    result: Annotated[OpResult, TensorExprType]

    def verify_(self):
        self.indices.verify_generic(Index | VectorSpace)
        if not matchTensorTupleStructures(self.expr.typ.result, self.indices, DeIndexing=True):
            raise Exception(f"IndexOp indicies do not match type of given expression")
        raise NotImplementedError
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
class SumOp(Operation):
    name : str = "dtl.sum"
    
    expr: Annotated[Operand, TensorExprType]
    indices: Annotated[Operand, ArrayAttr[Index]]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        for idx in self.indices:
            if idx not in self.expr.typ.indices:
                raise Exception(f"Sum op can only sum over indices that are arguments in the child expression")
        # if not matchTensorTupleStructures(self.expr.typ.result, self.indices, DeIndexing=True):
        #     raise Exception(f"IndexOp indicies do not match type of given expression")
        raise NotImplementedError


@irdl_op_definition
class ScalarAddOp(Operation):
    name : str = "dtl.scalarAdd"
    
    lhs: Annotated[Operand, TensorExprType]
    rhs: Annotated[Operand, TensorExprType]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        raise NotImplementedError


@irdl_op_definition
class ScalarMulOp(Operation):
    name : str = "dtl.scalarMul"
    
    lhs: Annotated[Operand, TensorExprType]
    rhs: Annotated[Operand, TensorExprType]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        raise NotImplementedError


@irdl_op_definition
class ScalarConstOp(Operation):
    name : str = "dtl.const"
    
    val: Annotated[Operand, builtin.AnyFloat]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        assert isa(self.val.typ, builtin.AnyFloat)
        assert len(self.result.typ.getIndices()) == 0, "dtl.const:: Type must have no indices"
        assert isa(self.result.typ.result, IndexShapeStruct[VectorSpace])
        assert len(self.result.typ.result.shape.data) == 0
        # raise NotImplementedError
        print("var COnst")
        
    @staticmethod
    def get(value: Union[Operation, SSAValue]) -> ScalarConstOp:
        result_type = TensorExprType.new([IndexToVectorSpaceMap.new([ArrayAttr([])]), IndexShapeStruct.new([ArrayAttr([])])])
        value = SSAValue.get(value)
        return ScalarConstOp.build(operands=[value], result_types=[result_type])


@irdl_op_definition
class TupleOp(Operation):
    name: str = "dtl.tuple"
    
    arguments: Annotated[VarOperand, TensorExprType]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        assert isa(self.result.typ, TensorExprType)
        assert isa(self.result.typ.result, IndexTupleStruct)
        raise NotImplementedError


@irdl_op_definition
class IndexedTupleOp(Operation):
    name: str = "dtl.indexedTuple"
    
    tuple: Annotated[Operand, TensorExprType]
    result: Annotated[OpResult, TensorExprType]
    
    def verify_(self):
        assert isa(self.tuple.typ, TensorExprType)
        assert isa(self.tuple.typ.result, IndexTupleStruct)
        raise NotImplementedError


#
#
# @irdl_attr_definition
# class TensorType(ParametrizedAttribute):
#     name : str = "dtl.tensor"
#     dims: ParameterDef[ArrayAttr[StringAttr | IntAttr]]
#
# @irdl_attr_definition
# class ScalarType(ParametrizedAttribute):
#     name : str = "dtl.scalar"

#
# @irdl_op_definition
# class LambdaOp(Operation):
#     name : str = "dtl.lambda"
#
#     inputs = VarOperandDef(TensorType)
#     return_type = AttributeDef(TensorType)
#     func_name = AttributeDef(StringAttr)
#     body = SingleBlockRegionDef()
#
#     def verify_(self):
#         ret = self.body.ops[-1]
#         if not isinstance(ret, LambdaYieldOp):
#             raise Exception(
#                 f"{LambdaYieldOp.name} expected as last operation of a {LambdaOp.name} node"
#             )
#         if ret.op.typ != self.return_type:
#             raise Exception(
#                 f"{LambdaOp.name} should have a {LambdaYieldOp.name} with the same return type"
#             )
#
#     def get_inputs(self):
#         return self.body.blocks[0].args
#
#
# @irdl_op_definition
# class LambdaYieldOp(Operation):
#     name : str = "dtl.return"
#
#     op = OperandDef(TensorType)
#
#     def verify_(self):
#         if not isinstance(self.parent.parent.parent, LambdaOp):
#             raise Exception(
#                 f"Parent of {LambdaYieldOp.name} should be a {LambdaOp.name}")

#
# @irdl_op_definition
# class IndexOp(Operation):
#     name : str = "dtl.index"
#
#     tensor: Annotated[Operand, TensorType]
#     indices = VarOperandDef(IndexType)
#     res = ResultDef(ScalarType)
#
#     def verify_(self):
#         if len(self.indices) != len(self.tensor.typ.dims.data):
#             raise Exception(
#                 f"An {IndexOp.name} should index a tensor with as many indices as its dimension"
#             )
#         for (idx, tensor_idx) in zip(self.indices, self.tensor.typ.dims.data):
#             if idx.typ.dim.data != tensor_idx.data:
#                 raise Exception(
#                     f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
#                 )

#
# @irdl_op_definition
# class DeIndexOp(Operation):
#     name : str = "dtl.deindex"
#
#     body = SingleBlockRegionDef()
#     res = ResultDef(TensorType)
#
#     def verify_(self):
#         if len(self.body.blocks[0].args) != len(self.res.typ.dims.data):
#             raise Exception(
#                 f"An {DeIndexOp.name} should return a tensor with as many dimensions as the index it produces"
#             )
#         for (idx, tensor_idx) in zip(self.body.blocks[0].args,
#                                      self.res.typ.dims.data):
#             if idx.typ.dim.data != tensor_idx.data:
#                 raise Exception(
#                     f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
#                 )
#
#         ret = self.body.ops[-1]
#         if not isinstance(ret, DeIndexYieldOp):
#             raise Exception(
#                 f"{DeIndexYieldOp.name} expected as last operation of a {DeIndexOp.name} node"
#             )
#
#     def get_ssa_indices(self):
#         return self.body.blocks[0].args
#
#
# @irdl_op_definition
# class DeIndexYieldOp(Operation):
#     name : str = "dtl.deindex_yield"
#
#     op = OperandDef(ScalarType)
#

#
# @irdl_op_definition
# class SumOp(Operation):
#     name : str = "dtl.sum"
#
#     body = SingleBlockRegionDef()
#     res = ResultDef(ScalarType)
#
#     def verify_(self):
#         if len(self.body.blocks[0].args) == 0:
#             raise Exception(
#                 f"A {SumOp.name} should sum over at least one index"
#             )
#         for idx in self.body.blocks[0].args:
#             if not isinstance(idx.typ, IndexType):
#                 raise Exception(f"A {SumOp.name} may only sum over an Indextype (args of enclosed Block)")
#         # if len(self.body.blocks[0].args) != len(self.res.typ.dims.data):
#         #     raise Exception(
#         #         f"A {SumOp.name} should return a tensor with as many dimensions as the index it produces"
#         #     )
#         # for (idx, tensor_idx) in zip(self.body.blocks[0].args,
#         #                              self.res.typ.dims.data):
#         #     if idx.typ.dim.data != tensor_idx.data:
#         #         raise Exception(
#         #             f"Index of size {idx.typ.dim.data} do not match with dimension of size {tensor_idx.data}"
#         #         )
#
#         ret = self.body.ops[-1]
#         if not isinstance(ret, SumYieldOp):
#             raise Exception(
#                 f"{SumYieldOp.name} expected as last operation of a {SumOp.name} node"
#             )
#
#     def get_ssa_indices(self):
#         return self.body.blocks[0].args

#
# @irdl_op_definition
# class SumYieldOp(Operation):
#     name : str = "dtl.sum_yield"
#
#     op = OperandDef(ScalarType)
#


# @irdl_op_definition
# class ScalarAddOp(Operation):
#     name : str = "dtl.scalarAdd"
#
#     lhs = OperandDef(ScalarType)
#     rhs = OperandDef(ScalarType)
#     res = ResultDef(ScalarType)
#
#
# @irdl_op_definition
# class ScalarMulOp(Operation):
#     name : str = "dtl.scalarMul"
#
#     lhs = OperandDef(ScalarType)
#     rhs = OperandDef(ScalarType)
#     res = ResultDef(ScalarType)


@dataclass
class DTL:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(IndexShapeStruct)
        self.ctx.register_attr(IndexTupleStruct)
        self.ctx.register_attr(Index)
        self.ctx.register_attr(KnownVectorSpace)
        self.ctx.register_attr(UnknownVectorSpace)
        self.ctx.register_attr(IndexToVectorSpaceMapPair)
        self.ctx.register_attr(IndexToVectorSpaceMap)
        # self.ctx.register_attr(TensorDimType)
        self.ctx.register_attr(TensorExprType)
        self.ctx.register_attr(NoneIndex)

        self.ctx.register_op(IndexBindingOp)
        self.ctx.register_op(IndexOp)
        self.ctx.register_op(DeIndexOp)
        self.ctx.register_op(SumOp)
        self.ctx.register_op(ScalarAddOp)
        self.ctx.register_op(ScalarMulOp)
        self.ctx.register_op(ScalarConstOp)
