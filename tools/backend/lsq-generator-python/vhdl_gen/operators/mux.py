from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *
from vhdl_gen.operators import *


# ===----------------------------------------------------------------------===#
# Multiplexer
# ===----------------------------------------------------------------------===#
# Mux1H    : One-hot select elements of `din` using `sel`
# Mux1HROM : Special multiplexer for the Group Allocator ROM.
# MuxIndex : Generate a VHDL array-index expression for selecting an element.
# MuxLookUp: Generate a conditional "when/else" lookup multiplexer in VHDL.

def Mux1H(ctx: VHDLContext, dout, din, sel, j=None) -> str:
    """
    Generate a one-hot multiplexer: for each element of "din", 
    write that bit/vector into a temporary and then OR-reduce into "dout".

    Parameters:
        dout (LogicVec or Logic):
            Destination for the multiplexed data, chosen by "sel".
        din (LogicVecArray or LogicArray or LogicVec):
            Source data.  
            - If LogicVecArray: 2D array of vectors.  
            - If LogicArray: 1D array of bits.  
            - If LogicVec: single vector.
        sel (LogicVec or LogicArray or LogicVecArray):
            One-hot select signals.
        j (int, optional):
            When "sel" is LogicVecArray, select the j-th "sel" signal.

    Returns:
        str: A VHDL code snippet for multiplexing.

    Example:
        type(din) = LogicVecArray:
          din = ("0010"; "1100"), sel = "10"
          -> dout = "0010"

        type(din) = LogicArray:
          din = ("0"; "1"; "1"), sel = "010"
          -> selects the middle bit: dout = '1'

        type(din) = LogicVec:
          din = "01101", sel = "00100"
          -> selects the third bit: dout = '1'
    """

    str_ret = ctx.get_current_indent() + '-- Mux1H Begin\n'
    str_ret += ctx.get_current_indent() + f'-- Mux1H({dout.name}, {din.name}, {sel.name})\n'
    ctx.use_temp()

    # din is always LogicVecArray
    if (type(din) == LogicVecArray):
        length = din.length
        size = din.size
        mux = LogicVecArray(ctx, ctx.get_temp('mux'), 'w', length, din.size)
    elif (type(din) == LogicArray):
        length = din.length
        size = None
        mux = LogicArray(ctx, ctx.get_temp('mux'), 'w', length)
    else:
        length = din.size
        size = None
        mux = LogicArray(ctx, ctx.get_temp('mux'), 'w', length)

    str_zero = Zero(size)
    if (j == None):
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(i)} <= {din.getNameRead(i)} ' +\
                f'when {sel.getNameRead(i)} = \'1\' else {str_zero};\n'
    else:
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(i)} <= {din.getNameRead(i)} ' +\
                f'when {sel.getNameRead(i, j)} = \'1\' else {str_zero};\n'

    str_ret += Reduce(ctx, dout, mux, 'or', False)
    str_ret += ctx.get_current_indent() + '-- Mux1H End\n\n'
    return str_ret


def Mux1HROM(ctx: VHDLContext, dout, din, sel, func=IntToBits) -> str:
    """
    Generate a one-hot ROM multiplexer for LSQ port index allocation,
    Load-Store Order Matrix construction, and tracking load/store numbers.

    Parameters:
        dout (LogicVecArray or LogicVec):
            If LogicVecArray: an NxM array; each row i will be computed independently.
                - ldq_port_idx_rom
                - stq_port_idx_rom
                - ga_ls_order_rom
            If LogicVec: a single M-bit vector; results from all groups are OR-reduced.
                - num_loads
                - num_stores

        din (list or list of lists): 
            ROM contents. (configs.gaLdPortIdx, configs.gaStPortIdx, configs.gaLdOrder
                           configs.gaNumLoads, configs. gaNumStores)

        sel (LogicArray):
            Indicates groups to be allocated. (group_init_hs)

        func (callable, optional):
            Conversion function from integer to LogicVec (default: IntToBits).
            Either IntToBits() or MaskLess()

    Behavior:
        - type(dout) == LogicVec:
            1. Build a temporary vector "mux" of width M.  
            2. For each group j, if sel[j] = '1', assign mux[j] <= func(din[j]);  
                else mux[j] <= Zero.  
            3. OR-reduce "mux" into the single "dout".

        - type(dout) == LogicVecArray:
            For each row i in dout:
                Repeat 1, 2, and 3.

    Example:
        Assume numBB = 3

        - type(dout) == LogicVec:
            dout = num_loads
            din  = configs.gaNumLoads = [3,1,2]
            sel  = "010"
            -> dout = "01" (1 load)

        This means that the currently allocated BB is BB1 (among BB0, BB1, and BB2)
        It has 1 load. that "dout" indicates 1.
    """

    str_ret = ctx.get_current_indent() + '-- Mux1H For Rom Begin\n'
    str_ret += ctx.get_current_indent() + f'-- Mux1H({dout.name}, {sel.name})\n'
    ctx.use_temp()
    mlen = sel.length
    size = dout.size
    str_zero = Zero(size)
    if (type(dout) == LogicVecArray):
        length = dout.length
        for i in range(0, length):
            str_ret += ctx.get_current_indent() + f'-- Loop {i}\n'
            mux = LogicVecArray(ctx, ctx.get_temp(f'mux_{i}'), 'w', mlen, size)
            for j in range(0, mlen):
                str_value = func(GetValue(din[j], i), size)
                if (str_value == str_zero):
                    str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(j)} <= {str_zero};\n'
                else:
                    str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(j)} <= {str_value} ' + \
                        f'when {sel.getNameRead(j)} else {str_zero};\n'
            str_ret += Reduce(ctx, dout[i], mux, 'or', False)
    else:   # type(dout) == LogicVec
        mux = LogicVecArray(ctx, ctx.get_temp(f'mux'), 'w', mlen, size)
        for j in range(0, mlen):
            str_value = func(din[j], size)
            if (str_value == str_zero):
                str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(j)} <= {str_zero};\n'
            else:
                str_ret += ctx.get_current_indent() + f'{mux.getNameWrite(j)} <= {str_value} ' + \
                    f'when {sel.getNameRead(j)} else {str_zero};\n'
        str_ret += Reduce(ctx, dout, mux, 'or', False)
    str_ret += ctx.get_current_indent() + '-- Mux1H For Rom End\n\n'
    return str_ret


def MuxIndex(din, sel) -> str:
    """
    Generate a VHDL array-index expression for selecting an element
    """
    return f'{din.getNameRead()}(to_integer(unsigned({sel.getNameRead()})))'


def MuxLookUp(ctx: VHDLContext, dout, din, sel) -> str:
    """
    Generate a conditional "when/else" lookup multiplexer in VHDL.

    Parameters:
        dout (Logic or LogicVec):
            Destination signal to receive the selected value.
        din (LogicArray or LogicVecArray):
            Array of input signals to choose from.
        sel (LogicVec):
            Binary select vector; compared against each index using IntToBits.

    Example:
        dout <= 
        din_0 when (sel = "0000") else
        din_1 when (sel = "0001") else
        din_2 when (sel = "0010") else
        din_3 when (sel = "0011") else
        din_4 when (sel = "0100") else
        din_5 when (sel = "0101") else
        din_6 when (sel = "0110") else
        din_7 when (sel = "0111") else
        din_8 when (sel = "1000") else
        din_9 when (sel = "1001") else
        '0';

        Depending on the value of "sel", "dout" is driven by the
        corresponding element of "din" or defaults to '0'.

    """

    str_ret = ctx.get_current_indent() + '-- MuxLookUp Begin\n'
    str_ret += ctx.get_current_indent() + f'-- MuxLookUp({dout.name}, {din.name}, {sel.name})\n'

    length = din.length
    size = sel.size
    str_ret += ctx.get_current_indent() + f'{dout.getNameWrite()} <= \n'

    for i in range(0, length):
        str_ret += ctx.get_current_indent() + f'{din.getNameRead(i)} ' +\
            f'when ({sel.getNameRead()} = {IntToBits(i, size)}) else\n'
    if (type(dout) == LogicVec):
        str_ret += ctx.get_current_indent() + f'{Zero(dout.size)};\n'
    else:
        str_ret += ctx.get_current_indent() + f'\'0\';\n'

    str_ret += ctx.get_current_indent() + '-- MuxLookUp End\n\n'
    return str_ret
