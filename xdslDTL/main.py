#!/usr/bin/env python3

import argparse

from xdsl.dialects.experimental.dlt import DLT
from xdsl.dialects.experimental.dtl import DTL
from xdsl.xdsl_opt_main import xDSLOptMain


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        super().register_all_dialects()
        self.ctx.load_dialect(DTL)
        self.ctx.load_dialect(DLT)

    def register_all_passes(self):
        super().register_all_passes()
        
    def register_all_targets(self):
        super().register_all_targets()


    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)


def __main__():
    xdsl_main = OptMain()
    xdsl_main.run()

if __name__ == "__main__":
    __main__()
