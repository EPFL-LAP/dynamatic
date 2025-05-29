from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *
from vhdl_gen.signals import *
from vhdl_gen.operators import *


def CyclicPriorityMasking(ctx: VHDLContext, dout, din, base, reverse=False) -> str:
    """
    Parameters:
        dout (LogicVecArray, LogicArray, LogicVec):
            Destination to write the masked result. 
            One youngest or oldest bit set to '1' and the other to '0' per each Array
        din  (LogicVecArray, LogicArray, LogicVec):
            Input data to be masked.
        base (LogicVec):
            Binary pivot index for the rotation mask.
        reverse (bool, optional): 
            Choose direction of masking.
            False -> Find the oldest   (Searching direction: base to MSB -> LSB to base)
            True  -> Find the youngest (Searching direction: base to LSB -> MSB to base)

    Example:
        1. din1 = 010110     2. din2 = 100100   3. din3 = 000110
           base = 001000        base = 001000      base = 001000   
           reverse = False      reverse = True     reverse = False  

           dout1= 010000        dout2= 000100      dout3= 000010
           (base to MSB)        (base to LSB)      (base to MSB -> LSB to base)

    Behavior (with the Example 1):
        double_in            = 010110 010110
        base                 = 000000 001000
        double_in - base     = 010110 001110
        ~(double_in - base)  = 101001 110001
        double_out           = double_in & ~(double_in - base)
                             = 000000 010000
        dout                 = 000000 | 010000 
                             = 010000
    """

    str_ret = ctx.get_current_indent() + '-- Priority Masking Begin\n'
    str_ret += ctx.get_current_indent() + f'-- CyclicPriorityMask({dout.name}, {din.name}, {base.name})\n'
    ctx.use_temp()
    if (type(din) == LogicVecArray):
        assert (reverse == False)
        for i in range(0, din.size):
            size = din.length
            double_in = LogicVec(ctx, ctx.get_temp(f'double_in_{i}'), 'w', size*2)
            for j in range(0, size):
                str_ret += Op(ctx, (double_in, j), (din, j, i))
                str_ret += Op(ctx, (double_in, j+size), (din, j, i))
            double_out = LogicVec(ctx, ctx.get_temp(f'double_out_{i}'), 'w', size*2)
            str_ret += Op(ctx, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
                          )
            for j in range(0, size):
                str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(j, i)} <= ' + \
                    f'{double_out.getNameRead(j)} or {double_out.getNameRead(j+size)};\n'
    else:
        if reverse:
            if (type(din) == LogicArray):
                size = din.length
            else:
                size = din.size
            double_in = LogicVec(ctx, ctx.get_temp('double_in'), 'w', size*2)
            for i in range(0, size):
                str_ret += Op(ctx, (double_in, i), (din, size-1-i))
                str_ret += Op(ctx, (double_in, i+size), (din, size-1-i))
            base_rev = LogicVec(ctx, ctx.get_temp('base_rev'), 'w', size)
            for i in range(0, size):
                str_ret += Op(ctx, (base_rev, i), (base, size-1-i))
            double_out = LogicVec(ctx, ctx.get_temp('double_out'), 'w', size*2)
            str_ret += Op(ctx, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base_rev, ')', ')'
                          )
            for i in range(0, size):
                str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(size-1-i)} <= ' + \
                    f'{double_out.getNameRead(i)} or {double_out.getNameRead(i+size)};\n'
        else:
            if (type(din) == LogicArray):
                size = din.length
                double_in = LogicVec(ctx, ctx.get_temp('double_in'), 'w', size*2)
                for i in range(0, size):
                    str_ret += Op(ctx, (double_in, i), (din, i))
                    str_ret += Op(ctx, (double_in, i+size), (din, i))
            else:
                size = din.size
                double_in = LogicVec(ctx, ctx.get_temp('double_in'), 'w', size*2)
                str_ret += Op(ctx, double_in, din, '&', din)
            double_out = LogicVec(ctx, ctx.get_temp('double_out'), 'w', size*2)
            str_ret += Op(ctx, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
                          )
            if (type(dout) == LogicVec):
                str_ret += ctx.get_current_indent() + f'{dout.getNameWrite()} <= ' + \
                    f'{double_out.getNameRead()}({size-1} downto 0) or ' + \
                    f'{double_out.getNameRead()}({2*size-1} downto {size});\n'
            else:
                for i in range(0, size):
                    str_ret += ctx.get_current_indent() + f'{dout.getNameWrite(i)} <= ' + \
                        f'{double_out.getNameRead(i)} or {double_out.getNameRead(i+size)};\n'
    str_ret += ctx.get_current_indent() + '-- Priority Masking End\n\n'
    return str_ret
