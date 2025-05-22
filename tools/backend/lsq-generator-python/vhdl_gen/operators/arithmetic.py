from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *


def WrapAdd(ctx: VHDLContext, out, in_a, in_b, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a + in_b
    else:
        "sum", "res" -> one extra bit to extend the bit-width
        Concatenates '0' to each input to extend the bit-width

        sum = in_a + in_b

        if sum >= max:
            out = sum - max
        else
            out = sum
    """

    str_ret = ctx.get_current_indent() + '-- WrapAdd Begin\n'
    str_ret += ctx.get_current_indent() + f'-- WrapAdd({out.name}, {in_a.name}, {in_b.name}, {max})\n'
    if (isPow2(max)):
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + unsigned({in_b.getNameRead()}));\n'
    else:
        ctx.use_temp()
        sum = LogicVec(ctx, ctx.get_temp('sum'), 'w', out.size + 1)
        res = LogicVec(ctx, ctx.get_temp('res'), 'w', out.size + 1)
        str_ret += ctx.get_current_indent() + f'{sum.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned(\'0\' & {in_a.getNameRead()}) + unsigned(\'0\' & {in_b.getNameRead()}));\n'
        str_ret += ctx.get_current_indent() + f'{res.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({sum.getNameRead()}) - {max}) ' + \
            f'when {sum.getNameRead()} >= {max} else {sum.getNameRead()};\n'
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= {res.getNameRead()}({out.size-1} downto 0);\n'
    str_ret += ctx.get_current_indent() + '-- WrapAdd End\n\n'
    return str_ret


def WrapAddConst(ctx: VHDLContext, out, in_a, const: int, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a + const
    else:
        if in_a + const >= max:
            out = in_a + const - max
        else:
            out = in_a + const
    """

    str_ret = ctx.get_current_indent() + '-- WrapAdd Begin\n'
    str_ret += ctx.get_current_indent() + f'-- WrapAdd({out.name}, {in_a.name}, {const}, {max})\n'
    if (isPow2(max)):
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + {const});\n'
    else:
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - {max - const}) ' + \
            f'when {in_a.getNameRead()} >= {max - const} else ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + {const}));\n'
    str_ret += ctx.get_current_indent() + '-- WrapAdd End\n\n'
    return str_ret


def WrapSub(ctx: VHDLContext, out, in_a, in_b, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a - in_b
    else:
        if in_a >= in_b:
            out = in_a - in_b
        else:
            out = (in_a + max) - in_b
    """

    str_ret = ctx.get_current_indent() + '-- WrapSub Begin\n'
    str_ret += ctx.get_current_indent() + f'-- WrapSub({out.name}, {in_a.name}, {in_b.name}, {max})\n'
    if (isPow2(max)):
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - unsigned({in_b.getNameRead()}));\n'
    else:
        str_ret += ctx.get_current_indent() + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - unsigned({in_b.getNameRead()})) ' + \
            f'when {in_a.getNameRead()} >= {in_b.getNameRead()} else\n' + '\t'*(ctx.tabLevel+1) + \
            f'std_logic_vector({max} - unsigned({in_b.getNameRead()}) + unsigned({in_a.getNameRead()}));\n'
    str_ret += ctx.get_current_indent() + '-- WrapAdd End\n\n'
    return str_ret
