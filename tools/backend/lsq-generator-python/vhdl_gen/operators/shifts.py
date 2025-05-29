from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *


# ===----------------------------------------------------------------------===#
# Cyclic Left Shift
# ===----------------------------------------------------------------------===#
# The following functions implement cyclic left shifts:
#   RotateLogicVec()      : Recursively shift a single vector.
#   RotateLogicArray()    : Recursively shift an array of single-bit elements.
#   RotateLogicVecArray() : Recursively shift an array of vectors.
#   -> These are called only internally by CyclicLeftShift().
#
# CyclicLeftShift():
#   Detects the type of `din` and dispatches to the appropriate implementation.

def RotateLogicVec(ctx: VHDLContext, dout, din, distance, layer) -> str:
    """
    Recursively perform a cyclic left shift of the vector "din" by the amount 
    specified in "distance".

    Parameters:
        dout     (LogicVec): Destination vector to hold the shifted result.
        din      (LogicVec): Source vector to be shifted.
        distance (LogicVec): Binary vector representing the shift amount.
        layer    (int)     : Current recursion layer; set to "distance.size-1" when called initially.

        The "layer" parameter is used internally to control recursion depth and
        should always start at "distance.size - 1".

    Returns:
        str_ret (str): A VHDL code snippet implementing the cyclic left shift.

    Usage:
        (Called only internally by CyclicLeftShift)
        RotateLogicVec(dout, din, distance, distance.size - 1)

        When this method is called, "layer" is always "distance.size - 1".
        "layer" is just for an recursive action.


    Example: 
        Input:  din  = "01110010", distance = 3
        Output: dout = "10010011"
    """

    str_ret = ''
    length = din.size
    if (layer == 0):
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        ctx.use_temp()
        res = LogicVec(ctx, ctx.get_temp('res'), 'w', length)
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += RotateLogicVec(ctx, dout, res, distance, layer-1)
    return str_ret


def RotateLogicArray(ctx: VHDLContext, dout, din, distance, layer) -> str:
    """
    Recursively perform a cyclic left shift of LogicArray "din" by the amount 
    specified in "distance".

    Identical in behavior to RotateLogicVec, but operates on multiple VHDL single-bit std_logic
    instead of std_logic_vector.

    """

    str_ret = ''
    length = din.length
    if (layer == 0):
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        ctx.use_temp()
        res = LogicArray(ctx, ctx.get_temp('res'), 'w', length)
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += RotateLogicArray(ctx, dout, res, distance, layer-1)
    return str_ret


def RotateLogicVecArray(ctx: VHDLContext, dout, din, distance, layer) -> str:
    """
    Recursively perform a cyclic left shift of the LogicVecArray "din" by the amount 
    specified in "distance".

    Identical in behavior to RotateLogicVec, but operates on multiple VHDL vectors std_logic_vector.
    For every LogicVec in LogicVecArray, cyclic left shift by "distance".

    Example:
        din = "11001001
               11100011"
        distance = 2

        -> dout = "00100111     (Cyclic Left Shift of each vector by 2)
                   10001111"
    """

    str_ret = ''
    length = din.length
    if (layer == 0):
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        ctx.use_temp()
        res = LogicVecArray(ctx, ctx.get_temp('res'), 'w', length, dout.size)
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer) % length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += RotateLogicVecArray(ctx, dout, res, distance, layer-1)
    return str_ret


def CyclicLeftShift(ctx: VHDLContext, dout, din, distance) -> str:
    """
    Execute a cyclic left shift operation based on the type of "din"

    This function wraps the three implementations:
        - RotateLogicVec        : when "din" is LogicVec
        - RotateLogicArray      : when "din" is LogicArray
        - RotateLogicVecArray   : when "din" is LogicVecArray

    Parameters:
        dout    : Destination signal to receive the shifted data.
        din     : Source data to be shifted.
        distance: Binary vector specifying how many positions to shift.

    Returns:
        str_ret : A VHDL code snippet (with indentation) implementing the cyclic left shift.
    """

    str_ret = ctx.get_current_indent() + '-- Shifter Begin\n'
    str_ret += ctx.get_current_indent() + f'-- CyclicLeftShift({dout.name}, {din.name}, {distance.name})\n'
    if (type(din) == LogicArray):
        str_ret += RotateLogicArray(ctx, dout, din, distance, distance.size-1)
    elif (type(din) == LogicVecArray):
        str_ret += RotateLogicVecArray(ctx, dout, din, distance, distance.size-1)
    else:
        str_ret += RotateLogicVec(ctx, dout, din, distance, distance.size-1)
    str_ret += ctx.get_current_indent() + '-- Shifter End\n\n'
    return str_ret
