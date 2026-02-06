from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *
from vhdl_gen.operators import *


# ===----------------------------------------------------------------------===#
# Reduction
# ===----------------------------------------------------------------------===#
# The following functions implement cyclic left shifts:
#   ReduceLogicVec()      : Recursively reduce a single vector.
#   ReduceLogicArray()    : Recursively reduce an array of single-bit elements.
#   ReduceLogicVecArray() : Recursively reduce an array of vectors.
#   -> These are called only internally by Reduce().
#
# Reduce():
#   Detects the type of `din` and dispatches to the appropriate implementation.

def ReduceLogicVec(ctx: VHDLContext, dout, din, operator, length) -> str:
    """
    Recursively reduce the vector "din" by "operator".

    Parameters:
        dout     (Logic)   : Destination std_logic to hold the reduced result.
        din      (LogicVec): Source vector to be reduced.
        operator (str)     : 'and', 'or', ...
        length   (int)     : Current recursion length;
                             set to "2**(log2Ceil(din.size) - 1)" when called initially.

        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

    Returns:
        str_ret (str): A VHDL code snippet implementing the LogicVec reduction.

    Usage:
        (Called only internally by Reduce)
        ReduceLogicVec(dout, din, operator, 2**(log2Ceil(din.size) - 1))

        When this method is called, "length" is always "2**(log2Ceil(din.size) - 1)".
        "length" is just for an recursive action.


    Example: 
        1. din = "01110010", operator = 'and' -> dout = '0'
        2. din = "01100111", operator = 'or'  -> dout = '1'
        3. din = "abcdefghijklmnop"
           dout = "a" operator "b" operator "c" operator "d" operator "e" operator "f"
                      operator "g" operator "h" operator "i" operator "j" operator "k"
                      operator "l" operator "m" operator "n" operator "o" operator "p" 
    """

    str_ret = ''
    if (length == 1):
        str_ret += ctx.get_current_indent() + f'{dout.getNameWrite()} <= ' + \
            f'{din.getNameRead(0)} {operator} {din.getNameRead(1)};\n'
    else:
        ctx.use_temp()
        res = LogicVec(ctx, ctx.get_temp('res'), 'w', length)
        for i in range(0, din.size - length):
            str_ret += ctx.get_current_indent() + f'{res.getNameWrite(i)} <= ' + \
                f'{din.getNameRead(i)} {operator} {din.getNameRead(i+length)};\n'
        for i in range(din.size - length, length):
            str_ret += ctx.get_current_indent() + f'{res.getNameWrite(i)} <= ' + \
                f'{din.getNameRead(i)};\n'
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += ReduceLogicVec(ctx, dout, res, operator, length//2)
    return str_ret


def ReduceLogicArray(ctx: VHDLContext, dout, din, operator, length) -> str:
    """
    Recursively perform reduction of LogicArray "din" by "operator".

    Identical in behavior to ReduceLogicVec, but operates on multiple VHDL single-bit std_logic
    instead of std_logic_vector.
    """

    str_ret = ''
    if (length == 1):
        str_ret += Op(ctx, dout, din[0], operator, din[1])
    else:
        ctx.use_temp()
        res = LogicArray(ctx, ctx.get_temp('res'), 'w', length)
        for i in range(0, din.length - length):
            str_ret += Op(ctx, res[i], din[i], operator, din[i+length])
        for i in range(din.length - length, length):
            str_ret += Op(ctx, res[i], din[i])
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += ReduceLogicArray(ctx, dout, res, operator, length//2)
    return str_ret


def ReduceLogicVecArray(ctx: VHDLContext, dout, din, operator, length) -> str:
    """
    Recursively perform reduction of the LogicVecArray "din" by "operator".

    Parameters:
        dout     (LogicVec)     : Destination std_logic_vector to hold the reduced result.
        din      (LogicVecArray): Source LogicVecArray to be reduced.
        operator (str)          : 'and', 'or', ...
        length   (int)          : Current recursion length;
                                  set to "2**(log2Ceil(din.size) - 1)" when called initially.

        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

    Returns:
        str_ret (str): A VHDL code snippet implementing the LogicVecArray reduction.

    Usage:
        (Called only internally by Reduce)
        ReduceLogicVecArray(dout, din, operator, 2**(log2Ceil(din.size) - 1))

        When this method is called, "length" is always "2**(log2Ceil(din.size) - 1)".
        "length" is just for an recursive action.

    Example:
        din = (LogicVecArray x with length of 8, each Vec size 16) where
        x[0]  = "a1 a2 a3 ... a16"
        x[1]  = "b1 b2 b3 ... b16"
        ...
        x[7]  = "p1 p2 p3 ... p16"

        dout = x[0] operator x[1] operator ... operator x[7]

        If operator = '&',
        dout = {a1 & b1 & ... & p1, a2 & b2 & ... & p2, ..., a16 & b16 & ... & p16}

        Therefore, dout is LogicVec.
    """
    str_ret = ''
    if (length == 1):
        str_ret += Op(ctx, dout, din[0], operator, din[1])
    else:
        ctx.use_temp()
        res = LogicVecArray(ctx, ctx.get_temp('res'), 'w', length, dout.size)
        for i in range(0, din.length - length):
            str_ret += Op(ctx, res[i], din[i], operator, din[i+length])
        for i in range(din.length - length, length):
            str_ret += Op(ctx, res[i], din[i])
        str_ret += ctx.get_current_indent() + '-- Layer End\n'
        str_ret += ReduceLogicVecArray(ctx, dout, res, operator, length//2)
    return str_ret


def Reduce(ctx: VHDLContext, dout, din, operator, comment: bool = True) -> str:
    """
    Execute reduction based on the type of "din"

    This function wraps the three implementations:
        - ReduceLogicVec        : when "din" is LogicVec
        - ReduceLogicArray      : when "din" is LogicArray
        - ReduceLogicVecArray   : when "din" is LogicVecArray

    Parameters:
        dout    : Destination signal to receive the reduced data.
        din     : Source data to be reduced.
        operator: types of operator for the reduction
        comment : Turn on/off adding VHDL comment lines.

    Returns:
        str_ret : A VHDL code snippet (with indentation) implementing the reduction.
    """

    str_ret = ''
    if (comment):
        str_ret += ctx.get_current_indent() + '-- Reduction Begin\n'
        str_ret += ctx.get_current_indent() + f'-- Reduce({dout.name}, {din.name}, {operator})\n'
    if (type(din) == LogicVec):
        if (din.size == 1):
            str_ret += Op(ctx, dout, (din, 0))
        else:
            length = 2**(log2Ceil(din.size) - 1)
            str_ret += ReduceLogicVec(ctx, dout, din, operator, length)
    else:
        if (din.length == 1):
            str_ret += Op(ctx, dout, din[0])
        else:
            length = 2**(log2Ceil(din.length) - 1)
            if (type(din) == LogicArray):
                str_ret += ReduceLogicArray(ctx, dout, din, operator, length)
            else:
                str_ret += ReduceLogicVecArray(ctx, dout, din, operator, length)
    if (comment):
        str_ret += ctx.get_current_indent() + '-- Reduction End\n\n'
    return str_ret
