#!/usr/bin/env python3

import argparse
# import ast
# import os
import sys
from io import StringIO, IOBase

# from xdslDTL.lower_to_python import lower_to_python
# from xdsl.parser import Parser
# from xdsl.printer import Printer
# from xdsl.ir import MLContext
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.memref import MemRef
# from xdsl.dialects.builtin import Builtin, ModuleOp
# from dialect import *
from transform import *

from typing import Dict, Callable, List

from xdsl.xdsl_opt_main import xDSLOptMain


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        super().register_all_dialects()
        dtl = DTL(self.ctx)

    def register_all_passes(self):
        super().register_all_passes()
        
    def register_all_targets(self):
        super().register_all_targets()
        def lower_to_python_func(prog: ModuleOp, output: IOBase):
            # string = lower_to_python(prog)
            string = "nah"
            output.write(string)
        self.available_targets['py'] = lower_to_python_func

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)


def __main__():
    print("main2")
    xdsl_main = OptMain()
    print("main3")
    xdsl_main.run()
    print("main4")


# if __name__ == "__main__":
#     print("main1")
#     __main__()
#
if __name__ == "__main__":
    ctx = MLContext()
    builtin = Builtin(ctx)
    func = Func(ctx)
    arith = Arith(ctx)
    memref = MemRef(ctx)
    affine = Affine(ctx)
    scf = Scf(ctx)
    dtl = DTL(ctx)

    f = sys.stdin
    input_str = f.read()
    parser = Parser(ctx, input_str)
    module = parser.parse_op()
    module.verify()
    if not (isinstance(module, ModuleOp)):
        raise Exception(
            "Expected module or program as toplevel operation")
    printer = Printer()
    printer.print_op(module)
