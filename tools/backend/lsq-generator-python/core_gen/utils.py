#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Helper functions shared by the generators.
# The signal classes live in core_gen/signals.py.
import math

# ===----------------------------------------------------------------------===#
# Helper Function
# ===----------------------------------------------------------------------===#


def MaskLess(din, size) -> str:
    """
    Example:
        MaskLess(3, 5)  # Output: "00111"
        MaskLess(2, 6)  # Output: "000011"
        MaskLess(5, 5)  # Output: "11111"
        MaskLess(0, 4)  # Output: "0000"
    """
    if (din > size):
        raise ValueError("Unknown value!")
    return '\"' + '0'*(size-din) + '1'*din + '\"'


def IntToBits(din, size=None) -> str:
    if size == None:
        if din == 1:
            return "'1'"
        elif din == 0:
            return "'0'"
        else:
            raise ValueError("IntToBits: Invalid value for size=None! Cannot represent the value as a single bit!")
    else:
        if din < 0:
            raise ValueError("IntToBits: Negative value cannot be converted to bits!")
        if din >= (1 << size):
            raise ValueError(f"IntToBits: Value {din} cannot be represented with {size} bit(s)!")
        bits = f"{din:0{size}b}"
        return f'"{bits}"'


def Zero(size) -> str:
    if size == None:
        return "'0'"
    else:
        return '"' + "0" * size + '"'


def GetValue(row, i) -> int:
    if len(row) > i:
        return row[i]
    else:
        return 0


def isPow2(value: int) -> bool:
    return (value & (value-1) == 0) and value != 0


def log2Ceil(value: int) -> int:
    return math.ceil(math.log2(value))
