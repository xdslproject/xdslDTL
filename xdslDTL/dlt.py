from xdsl.irdl import irdl_op_definition, irdl_attr_definition, ParametrizedAttribute, ParameterDef, Attribute, IRDLOperation, operand_def, result_def
import xdsl.dialects.builtin as builtin
from typing import TypeAlias
from xdsl.ir.core import Dialect
from xdsl.utils.hints import isa



@irdl_attr_definition
class DimensionType(ParametrizedAttribute):
    name = "dlt.dimensionType"
    dim_names: ParameterDef[builtin.ArrayAttr[builtin.StringAttr]]
    child_type: ParameterDef[builtin.TypeAttribute]

    def verify(self) -> None:
        names = [dim.data for dim in self.dim_names.data]
        assert names == sorted(names), "dlt.dimensionType: dim names must be alphabetically sorted"

@irdl_attr_definition
class StructField(ParametrizedAttribute):
    name = "dlt.structField"
    field_name: ParameterDef[builtin.StringAttr]
    field_type: ParameterDef[DimensionType | builtin.TypeAttribute]


@irdl_attr_definition
class StructType(ParametrizedAttribute):
    name = "dlt.structType"
    struct_name: ParameterDef[builtin.StringAttr]
    fields: ParameterDef[builtin.ArrayAttr[StructField]]
    # child_names: ParameterDef[builtin.ArrayAttr[builtin.StringAttr]]
    # child_types: ParameterDef[builtin.ArrayAttr[DimensionType]]

    def verify(self) -> None:
        names = [child.field_name.data for child in self.fields]
        assert names == sorted(names), "dlt.structType: fields must be sorted alphabetically"
        assert len(names) == len(set(names)), "dlt.structType: duplicate names found"

        # fields look like 'bot.Robot:pos.Vec3:x'
        # check that all instances of a struct name have the same fields
        # make a map of struct name to fields found
        struct_map = {}
        struct_names = set()
        for field in self.fields:
            field_name = field.field_name.data
            parts = field_name.split('.')
            for part in parts:
                s_name, s_field = part.split(':',1)
                struct_names += s_name
                if s_name not in struct_map:
                    struct_map[s_name] = set()
                struct_map[s_name] += s_field
        for struct in struct_names:
            ident = struct + ':'
            fields = [f.field_name.data for f in self.fields if ident in f.field_name.data]
            parent_map = {}
            for field in fields:
                parts = field.split(ident)
                assert len(parts) == 2, "dlt.structType: recursive structures are not allowed"
                context = parts[0]
                if context not in parent_map:
                    parent_map[context] = set()
                parent_map[context] += parts[1].split('.')[0]
            for context, field_names in parent_map.items():
                assert field_names != struct_map[struct], f"dlt.structType: multiple definitions of {struct} found."




        pass


@irdl_attr_definition
class DataTreeType(ParametrizedAttribute):
    name = "dlt.dataTreeType"
    # args: ParameterDef[]
    # result: ParameterDef[]

    def verify(self) -> None:
        pass


@irdl_attr_definition
class DenseType(ParametrizedAttribute):
    name = "dlt.dense"
    dimension: ParameterDef[builtin.StringAttr]
    child: ParameterDef[Attribute]

    def verify(self) -> None:
        isa(self.child, dltType)
        pass


dlt_i_Type: TypeAlias = DenseType
dltType: TypeAlias = dlt_i_Type | builtin.Float32Type | builtin.Float64Type | builtin.IntegerType | builtin.IndexType

@irdl_op_definition
class DataOp(IRDLOperation):
    name = "dlt.data"

    data = operand_def()
    result: DataTreeType = result_def(DataTreeType)

    def verify_(self):
        pass


@irdl_op_definition
class GetterOp(IRDLOperation):
    name = "dlt.get"

    data_val: DataTreeType = operand_def(DataTreeType)
    result: DataTreeType = result_def(DataTreeType)

    def verify_(self):
        pass


DLT = Dialect(
    [
        GetterOp
    ],
    [
        DimensionType,
        StructField,
        StructType,
        DataTreeType,
        DenseType,
    ],
)