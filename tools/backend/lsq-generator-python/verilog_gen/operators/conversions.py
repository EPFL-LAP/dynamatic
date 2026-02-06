from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *
from vhdl_gen.operators import *


def VecToArray(ctx: VHDLContext, dout, din) -> str:
    """
    Converts LogicVec to LogicArray

    Parameter:
        dout (LogicArray)
        din  (LogicVec)

    Example:
        din = "0101"
        dout = ('0'; '1'; '0'; '1')
    """
    size = din.size
    assert dout.length == size
    str_ret = ''
    for i in range(0, size):
        str_ret += Op(ctx, (dout, i), (din, i))
    return str_ret


def BitsToOH(ctx: VHDLContext, dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.

    Example:
        din  = "01"
        dout = "0010"
    """
    str_ret = ctx.get_current_indent() + '-- Bits To One-Hot Begin\n'
    str_ret += ctx.get_current_indent() + f'-- BitsToOH({dout.name}, {din.name})\n'
    for i in range(0, dout.size):
        str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= ' \
            f'\'1\' when {din.getNameRead()} = {IntToBits(i, din.size)} else \'0\';\n'
    str_ret += ctx.get_current_indent() + '-- Bits To One-Hot End\n\n'
    return str_ret


def BitsToOHSub1(ctx: VHDLContext, dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.
    The result one-hot representation should be cyclic right shifted.

    Example:
        din  = "01"
        dout = "0001"
    """
    str_ret = ctx.get_current_indent() + '-- Bits To One-Hot Begin\n'
    str_ret += ctx.get_current_indent() + f'-- BitsToOHSub1({dout.name}, {din.name})\n'
    for i in range(0, dout.size):
        str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= ' \
            f'\'1\' when {din.getNameRead()} = {IntToBits((i+1) % dout.size, din.size)} else \'0\';\n'
    str_ret += ctx.get_current_indent() + '-- Bits To One-Hot End\n\n'
    return str_ret


def OHToBits(ctx: VHDLContext, dout, din) -> str:
    """
    Generate VHDL code to convert a one-hot vector into its binary index.

    Example:
        din  = "0010"
        dout = "01"
    """

    str_ret = ctx.get_current_indent() + '-- One-Hot To Bits Begin\n'
    str_ret += ctx.get_current_indent() + f'-- OHToBits({dout.name}, {din.name})\n'
    size = dout.size
    size_in = din.size
    ctx.use_temp()
    for i in range(0, size):
        temp_in = LogicArray(ctx, ctx.get_temp(f'in_{i}'), 'w', size_in)
        temp_out = Logic(ctx, ctx.get_temp(f'out_{i}'), 'w')
        for j in range(0, size_in):
            if ((j // (2**i)) % 2 == 1):
                str_ret += Op(ctx, (temp_in, j), (din, j))
            else:
                str_ret += Op(ctx, (temp_in, j), '\'0\'')
        str_ret += Reduce(ctx, temp_out, temp_in, 'or', False)
        str_ret += Op(ctx, (dout, i), temp_out)
    str_ret += ctx.get_current_indent() + '-- One-Hot To Bits End\n\n'
    return str_ret
