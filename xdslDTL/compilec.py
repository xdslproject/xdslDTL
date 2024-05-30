import os
import subprocess
import tempfile
from io import StringIO

from nptyping import Shape, NDArray
import numpy as np

from xdsl import ir
from xdsl.dialects import builtin, memref, func
from xdsl.printer import Printer



def compile(module: builtin.ModuleOp, lib_output: str, header_out=None):
    # if header_out==None:
    #     header_out = lib_output.removesuffix(".o") + ".h"

    print(f"Compile to Binary: {lib_output}")

    print("Module:")
    print(module)
    # print("args")
    # print(func.args)
    # print("results")
    # print(func.get_return_op())
    # print("func type")
    # print(func.function_type)
    #
    print("mlir output:")
    res = StringIO()
    printer = Printer(print_generic_format=False, stream=res)
    printer.print(module)
    print(res.getvalue())

    fd, path = tempfile.mkstemp()
    print(f"Making tmp mlir - IR file: {path}")
    with os.fdopen(fd, 'wb') as tmp:
        tmp.write(res.getvalue().encode('utf8'))



    print("mlir-opt:")
    passes = [
        "--convert-math-to-funcs",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--expand-strided-metadata",
        "--normalize-memrefs",
        "--memref-expand",
        "--fold-memref-alias-ops",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    print("command:")
    print(' '.join(['mlir-opt'] + passes))
    process_opt = subprocess.Popen(['mlir-opt'] + passes, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process_opt.communicate(res.getvalue().encode('utf8'))
    process_opt.wait()
    print("stdout:")
    print(out.decode('utf8'))
    print("stderr:")
    print(err.decode('utf8') if err is not None else None)

    print("mlir-translate")
    process_translate = subprocess.Popen(['mlir-translate', '--mlir-to-llvmir'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process_translate.communicate(out)
    process_translate.wait()
    print("stdout:")
    print(out.decode('utf8'))
    print("stderr:")
    print(err.decode('utf8') if err is not None else None)
    fd, path = tempfile.mkstemp(suffix=".ll")
    print(f"Making tmp llvm-IR file: {path}")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(out)

            clang_args = ["clang"]
            clang_args.extend([ '-o', lib_output])
            clang_args.append('-shared')
            # clang_args.append("-c")
            # clang_args.append("-v")
            clang_args.append("-g")
            clang_args.append("-O3")
            clang_args.append(path)


            print(" ".join(clang_args))
            process_clang = subprocess.Popen(clang_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # out, err = process_clang.communicate(out)
            # print("stdout:")
            # print(out.decode('utf8') if out is not None else None)
            # print("stderr:")
            # print(err.decode('utf8') if err is not None else None)
            process_clang.wait()

    finally:
        # os.remove(path)
        pass

    print("done")



