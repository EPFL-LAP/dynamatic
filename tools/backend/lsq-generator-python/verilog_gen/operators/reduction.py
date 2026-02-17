from verilog_gen.emitters import Emitter
from verilog_gen.utils import *
from verilog_gen.operators import *
from verilog_gen.ir import Val, Bin


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

def ReduceLogicVec(em: Emitter, dout, din, operator, length) -> str:
    """
    Recursively reduce the vector "din" by "operator" and add this to "em".

    Parameters:
        dout     (Logic)   : Destination std_logic to hold the reduced result.
        din      (LogicVec): Source vector to be reduced.
        operator (str)     : 'and', 'or', ...
        length   (int)     : Current recursion length;
                             set to "2**(log2Ceil(din.size) - 1)" when called initially.

        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

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
    from verilog_gen.signals import LogicVec

    if (length == 1):
        em.add_assignment(dout, Bin(Val(din, 0), operator, Val(din, 1)))
    else:
        em.use_temp()
        res = LogicVec(em, em.get_temp('res'), 'w', length)
        for i in range(0, din.size - length):
            em.add_assignment((res, i), Bin(Val(din, i), operator, Val(din, i+length)))
        for i in range(din.size - length, length):
            em.add_assignment((res, i), Val(din, i))
        em.add_comment('Layer End')
        ReduceLogicVec(em, dout, res, operator, length//2)


def ReduceLogicArray(em: Emitter, dout, din, operator, length) -> str:
    """
    Recursively perform reduction of LogicArray "din" by "operator".

    Identical in behavior to ReduceLogicVec, but operates on multiple VHDL single-bit std_logic
    instead of std_logic_vector.
    """
    from verilog_gen.signals import LogicArray

    if (length == 1):
        em.add_assignment(dout, Bin(din[0], operator, din[1]))
    else:
        em.use_temp()
        res = LogicArray(em, em.get_temp('res'), 'w', length)
        for i in range(0, din.length - length):
            em.add_assignment(res[i], Bin(din[i], operator, din[i+length]))
        for i in range(din.length - length, length):
            em.add_assignment(res[i], din[i])
        em.add_comment('Layer End')
        ReduceLogicArray(em, dout, res, operator, length//2)


def ReduceLogicVecArray(em: Emitter, dout, din, operator, length) -> str:
    """
    Recursively perform reduction of the LogicVecArray "din" by "operator" and add this to "em".

    Parameters:
        dout     (LogicVec)     : Destination std_logic_vector to hold the reduced result.
        din      (LogicVecArray): Source LogicVecArray to be reduced.
        operator (str)          : 'and', 'or', ...
        length   (int)          : Current recursion length;
                                  set to "2**(log2Ceil(din.size) - 1)" when called initially.

        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

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
    from verilog_gen.signals import LogicVecArray, LogicArray
    if (length == 1):
        em.add_assignment(dout, Bin(din[0], operator, din[1]))
    else:
        em.use_temp()
        res = LogicVecArray(em, em.get_temp('res'), 'w', length, dout.size)
        for i in range(0, din.length - length):
            em.add_assignment(res[i], Bin(din[i], operator, din[i+length]))
        for i in range(din.length - length, length):
            em.add_assignment(res[i], din[i])
        em.add_comment('Layer End')
        ReduceLogicVecArray(em, dout, res, operator, length//2)


def Reduce(em: Emitter, dout, din, operator, comment: bool = True) -> str:
    """
    Execute reduction based on the type of "din" and add this to "em".

    This function wraps the three implementations:
        - ReduceLogicVec        : when "din" is LogicVec
        - ReduceLogicArray      : when "din" is LogicArray
        - ReduceLogicVecArray   : when "din" is LogicVecArray

    Parameters:
        dout    : Destination signal to receive the reduced data.
        din     : Source data to be reduced.
        operator: types of operator for the reduction
        comment : Turn on/off adding VHDL comment lines.
    """
    from verilog_gen.signals import LogicVec, LogicArray, LogicVecArray

    if (comment):
        em.add_comment('Reduction Begin')
        em.add_comment(f'Reduce({dout.name}, {din.name}, {em.get_binop_str(operator)})')
    if (type(din) == LogicVec):
        if (din.size == 1):
            em.add_assignment(dout, Val(din, 0))
        else:
            length = 2**(log2Ceil(din.size) - 1)
            ReduceLogicVec(em, dout, din, operator, length)
    else:
        if (din.length == 1):
            em.add_assignment(dout, Val(din[0]))
        else:
            length = 2**(log2Ceil(din.length) - 1)
            if (type(din) == LogicArray):
                ReduceLogicArray(em, dout, din, operator, length)
            else:
                ReduceLogicVecArray(em, dout, din, operator, length)
    if (comment):
        em.add_comment('Reduction End\n')