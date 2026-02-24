#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import re
import math

# ===----------------------------------------------------------------------===#
# Helper Function
# ===----------------------------------------------------------------------===#

def GetValue(row, i) -> int:
    if len(row) > i:
        return row[i]
    else:
        return 0


def isPow2(value: int) -> bool:
    return (value & (value-1) == 0) and value != 0


def log2Ceil(value: int) -> int:
    return math.ceil(math.log2(value))
