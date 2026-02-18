from verilog_gen.emitters import Emitter
from verilog_gen.ir import Op, Val, Bit
from verilog_gen.utils import *
from verilog_gen.signals import *


def WrapAdd(em: Emitter, out, in_a, in_b, max: int) -> str:
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

    em.add_comment('WrapAdd Begin')
    em.add_comment(f'WrapAdd({out.name}, {in_a.name}, {in_b.name}, {max})')
    if (isPow2(max)):
        em.add_assignment(out, in_a + in_b)
    else:
        em.use_temp()
        sum = LogicVec(em, em.get_temp('sum'), 'w', out.size + 1)
        res = LogicVec(em, em.get_temp('res'), 'w', out.size + 1)
        em.add_assignment(sum, Bit(0).concat(in_a) + Bit(0).concat(in_b))
        em.add_assignment(res, (sum - Val(max)).when(sum >= Val(max)).else_(sum))
        em.add_assignment(out, em.slice_var(res.getNameRead(), out_size-1, 0))
    em.add_comment("WrapAdd End\n")


def WrapAddConst(em: Emitter, out, in_a, const: int, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a + const
    else:
        if in_a + const >= max:
            out = in_a + const - max
        else:
            out = in_a + const
    """

    em.add_comment('WrapAdd Begin')
    em.add_comment(f'WrapAdd({out.name}, {in_a.name}, {const}, {max})')

    if (isPow2(max)):
        em.add_assignment(out, in_a + Val(const))
    else:
        em.add_assignment(out, (in_a - Val(max - const)).when(in_a >= Val(max - const)).else_(in_a + Val(const)))
    em.add_comment('WrapAdd End')


def WrapSub(em: Emitter, out, in_a, in_b, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a - in_b
    else:
        if in_a >= in_b:
            out = in_a - in_b
        else:
            out = (in_a + max) - in_b
    """

    em.add_comment('WrapSub Begin')
    em.add_comment(f'WrapSub({out.name}, {in_a.name}, {in_b.name}, {max})')
    if (isPow2(max)):
        em.add_assignment(out, in_a - in_b)
    else:
        em.add_assignment(out, (in_a - in_b).when(in_a >= in_b).else_(in_a + Val(max) - in_b))
    em.add_comment('WrapSub End')