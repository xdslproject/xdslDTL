#!/usr/bin/env python3

import argparse
import ast
import os
import sys
from io import StringIO

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.ir import MLContext
from xdsl.dialects.std import Std
from xdsl.dialects.scf import Scf
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.memref import MemRef
from xdsl.dialects.builtin import Builtin, ModuleOp
from dialect import *
from transform import *

from typing import Dict, Callable, List


if __name__ == "__main__":
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
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
