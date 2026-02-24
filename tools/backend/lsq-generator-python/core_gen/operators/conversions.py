from core_gen.signals import *
from core_gen.ir import Val, Bit, BinOp
from core_gen.emitters import Emitter
from core_gen.operators import Reduce


def VecToArray(em: Emitter, dout, din) -> str:
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
    for i in range(0, size):
        em.add_assignment((dout, i), Val(din, i))


def BitsToOH(em: Emitter, dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.

    Example:
        din  = "01"
        dout = "0010"
    """
    em.add_comment('Bits To One-Hot Begin')
    em.add_comment(f'BitsToOH({dout.name}, {din.name})')
    for i in range(0, dout.size):
        em.add_assignment((dout, i), Bit(1).when(din == Val(i, size=din.size)).else_(Bit(0)))
    em.add_comment('Bits To One-Hot End\n')


def BitsToOHSub1(em: Emitter, dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.
    The result one-hot representation should be cyclic right shifted.

    Example:
        din  = "01"
        dout = "0001"
    """
    em.add_comment('Bits To One-Hot Begin')
    em.add_comment(f'BitsToOHSub1({dout.name}, {din.name})')
    for i in range(0, dout.size):
        em.add_assignment((dout, i), Bit(1).when(din == Val((i+1) % dout.size, size=din.size)).else_(Bit(0)))
    em.add_comment('Bits To One-Hot End\n')


def OHToBits(em: Emitter, dout, din) -> str:
    """
    Generate VHDL code to convert a one-hot vector into its binary index.

    Example:
        din  = "0010"
        dout = "01"
    """

    em.add_comment('One-Hot To Bits Begin')
    em.add_comment(f'OHToBits({dout.name}, {din.name})')
    size = dout.size
    size_in = din.size
    em.use_temp()
    for i in range(0, size):
        temp_in = LogicArray(em, em.get_temp(f'in_{i}'), 'w', size_in)
        temp_out = Logic(em, em.get_temp(f'out_{i}'), 'w')
        for j in range(0, size_in):
            if ((j // (2**i)) % 2 == 1):
                em.add_assignment((temp_in, j), Val(din, j))
            else:
                em.add_assignment((temp_in, j), Bit(0))
        Reduce(em, temp_out, temp_in, BinOp.OR, False)
        em.add_assignment((dout, i), temp_out)
    em.add_comment('One-Hot To Bits End\n')
