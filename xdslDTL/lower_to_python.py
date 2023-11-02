from dtl import *
from dataclasses import dataclass, field
from typing import List, Dict

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.experimental.dtl import *
from xdsl.ir import SSAValue, Operation


def lower_to_python(module: ModuleOp) -> str:
    lowering = LowerToPython()
    return lowering.lower_module(module)

@dataclass
class LowerToPython:
    names: Dict[SSAValue, str] = field(default_factory=dict)
    new_name: int = field(default=0)
    indent: int = field(default=0)
    indent_size: int = field(default=2)


    def get_new_name(self, name) -> str:
        self.new_name += 1
        return f"{name}{self.new_name}"

    def new_line(self) -> str:
        return "\n" + " " * self.indent * self.indent_size

    def lower_module(self, module: ModuleOp):
        res = ""
        for op in module.ops:
            if isinstance(op, LambdaOp):
                res += self.lower_lambda(op)
        return res

    def lower_lambda(self, lam: LambdaOp) -> str:
        for arg in lam.body.blocks[0].args:
            self.names[arg] = self.get_new_name("INPUT")
        res = f"def {lam.func_name.data}("
        res += ",".join([self.names[inp] for inp in lam.get_inputs()])
        res += "):"
        self.indent += 1
        res += self.new_line()
        for op in lam.body.ops:
            res += self.lower_op(op)
        return res

    def new_array(self, dimensions: List[str]) -> str:
        assert len(dimensions) != 0
        if len(dimensions) == 1:
            return f"[0 for _ in range({dimensions[0]})]"
        return f"[{self.new_array(dimensions[1:])} for _ in range({dimensions[0]})]"

    def lower_deindex(self, deindex: DeIndexOp) -> str:
        # Allocate the new tensor
        new_name = self.get_new_name("DEINDEX")
        self.names[deindex.res] = new_name
        indices = [idx.typ.dim.data for idx in deindex.get_ssa_indices()]
        res = f"{new_name} = {self.new_array(indices)}"
        res += self.new_line()
        old_indent = self.indent

        # for loops to assign the indices
        for idx in deindex.get_ssa_indices():
            idx_name = self.get_new_name("i")
            self.names[idx] = idx_name
            res += f"for {idx_name} in range({idx.typ.dim.data}):"
            self.indent += 1
            res += self.new_line()

        for op in deindex.body.ops:
            res += self.lower_op(op)

        self.indent = old_indent
        return res
    

    def lower_sum(self, sum: SumOp) -> str:
        # Allocate the new tensor
        new_name = self.get_new_name("SUM")
        self.names[sum.res] = new_name
        indices = [idx.typ.dim.data for idx in sum.get_ssa_indices()]
        res = f"{new_name} = 0"
        res += self.new_line()
        old_indent = self.indent

        # for loops to assign the indices
        for idx in sum.get_ssa_indices():
            idx_name = self.get_new_name("s")
            self.names[idx] = idx_name
            res += f"for {idx_name} in range({idx.typ.dim.data}):"
            self.indent += 1
            res += self.new_line()

        for op in sum.body.ops:
            res += self.lower_op(op)

        self.indent = old_indent
        return res

    def lower_index(self, op: IndexOp) -> str:
        self.names[op.res] = self.get_new_name("index")
        res = f"{self.names[op.res]} = "
        res += f"{self.names[op.tensor]}"
        for idx in op.indices:
            res += f"[{self.names[idx]}]"
        res += self.new_line()
        return res
    
    def lower_scalar_add(self, op: ScalarAddOp) -> str:
        self.names[op.res] = self.get_new_name("add")
        res = f"{self.names[op.res]} = "
        res += f"{self.names[op.lhs]}"
        res += f"+"
        res += f"{self.names[op.rhs]}"
        res += self.new_line()
        return res
    
    def lower_scalar_mul(self, op: ScalarMulOp) -> str:
        self.names[op.res] = self.get_new_name("mul")
        res = f"{self.names[op.res]} = "
        res += f"{self.names[op.lhs]}"
        res += f"*"
        res += f"{self.names[op.rhs]}"
        res += self.new_line()
        return res

    def lower_deindex_yield(self, op: DeIndexYieldOp) -> str:
        deindex_parent = op.parent.parent.parent
        res = self.names[deindex_parent.res]
        indices =  deindex_parent.get_ssa_indices()
        for index in indices:
            res += f"[{self.names[index]}]"
        res += f" = {self.names[op.op]}"
        self.indent -= len(deindex_parent.get_ssa_indices())
        return res + self.new_line()
    
    def lower_sum_yield(self, op: SumYieldOp) -> str:
        sum_parent = op.parent.parent.parent
        res = self.names[sum_parent.res]
        # indices =  sum_parent.get_ssa_indices()
        # for index in indices:
        #     res += f"[{self.names[index]}]"
        res += f" += {self.names[op.op]}"
        self.indent -= len(sum_parent.get_ssa_indices())
        return res + self.new_line()

    def lower_op(self, op: Operation) -> str:
        if isinstance(op, DeIndexOp):
            return self.lower_deindex(op)
        if isinstance(op, LambdaYieldOp):
            return f"return {self.names[op.op]}"
        if isinstance(op, IndexOp):
            return self.lower_index(op)
        if isinstance(op, DeIndexYieldOp):
            return self.lower_deindex_yield(op)
        if isinstance(op, ScalarAddOp):
            return self.lower_scalar_add(op)
        if isinstance(op, ScalarMulOp):
            return self.lower_scalar_mul(op)
        if isinstance(op, SumOp):
            return self.lower_sum(op)
        if isinstance(op, SumYieldOp):
            return self.lower_sum_yield(op)
        return f"error {op.name}" + self.new_line()

