from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *


def Op(ctx: VHDLContext, out, *list_in) -> str:
    """
    Generates a proper VHDL assignment statement.

    Args:
        out: LHS of the assignment
        *list_in: A sequence of RHS elements

    Example: Op(ctx, valid, a,'when', b, 'else', 0)
            valid <= a when b else '0'

    """
    if type(out) == tuple:
        if len(out) == 2:
            str_ret = ctx.get_current_indent() + f'{out[0].getNameWrite(out[1])} <='
        else:
            str_ret = ctx.get_current_indent() + f'{out[0].getNameWrite(out[1], out[2])} <='
    else:
        str_ret = ctx.get_current_indent() + f'{out.getNameWrite()} <='
        if (type(out) == Logic):
            size = 1
        else:
            size = out.size
    for arg in list_in:
        if type(arg) == str:
            str_ret += ' ' + arg
        elif type(arg) == int:
            str_ret += ' ' + IntToBits(arg, size)
        elif type(arg) == tuple:
            if type(arg[0]) == int:
                str_ret += ' ' + IntToBits(arg[0], arg[1])
            elif len(arg) == 2:
                str_ret += ' ' + arg[0].getNameRead(arg[1])
            else:
                str_ret += ' ' + arg[0].getNameRead(arg[1], arg[2])
        else:
            str_ret += ' ' + arg.getNameRead()
    str_ret += ';\n'
    return str_ret
