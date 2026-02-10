from verilog_gen.emitters.emitter import Emitter
from verilog_gen.utils import *
from verilog_gen.signals import *
from verilog_gen.operators import *


def CyclicPriorityMasking(em: Emitter, dout, din, base, reverse=False) -> str:
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

    Example (LogicVecArray din):
        1. din = 010
                 000
                 100
                 010
           base = 001
           reverse = False

           priority masking -> (0th col) [0010] with base = 1 -> 0010
           priority masking -> (1st col) [1001] with base = 1 -> 0001
           priority masking -> (2nd col) [0000] with base = 1 -> 0000

           -> dout = 000
                     000
                     100
                     010
    """

    em.add_comment('Priority Masking Begin')
    em.add_comment(f'CyclicPriorityMask({dout.name}, {din.name}, {base.name})')
    em.use_temp()
    if (type(din) == LogicVecArray):
        assert (reverse == False)
        for i in range(0, din.size):
            size = din.length
            double_in = LogicVec(em, em.get_temp(f'double_in_{i}'), 'w', size*2)
            for j in range(0, size):
                em.add_assignment(Op(em, (double_in, j), (din, j, i)))
                em.add_assignment(Op(em, (double_in, j+size), (din, j, i)))
            double_out = LogicVec(em, em.get_temp(f'double_out_{i}'), 'w', size*2)
            em.add_assignment(Op(em, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
                          ))
            for j in range(0, size):
                em.add_assignment(Op(em, (dout, j, i), (double_out, j), 'or' (double_out, j+size)))
    else:
        if reverse:
            if (type(din) == LogicArray):
                size = din.length
            else:
                size = din.size
            double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
            for i in range(0, size):
                em.add_assignment(Op(em, (double_in, i), (din, size-1-i)))
                em.add_assignment(Op(em, (double_in, i+size), (din, size-1-i)))
            base_rev = LogicVec(em, em.get_temp('base_rev'), 'w', size)
            for i in range(0, size):
                str_ret += Op(em, (base_rev, i), (base, size-1-i))
            double_out = LogicVec(em, em.get_temp('double_out'), 'w', size*2)
            em.add_assignment(Op(em, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base_rev, ')', ')'
                          ))
            for i in range(0, size):
                em.add_assignment(Op(em, (dout, size-1-i), (double_out, i), 'or', (double_out, i+size)))
        else:
            if (type(din) == LogicArray):
                size = din.length
                double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
                for i in range(0, size):
                    em.add_assignment(Op(em, (double_in, i), (din, i)))
                    em.add_assignment(Op(em, (double_in, i+size), (din, i)))
            else:
                size = din.size
                double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
                em.add_assignment(Op(em, double_in, din, '&', din))
            double_out = LogicVec(em, em.get_temp('double_out'), 'w', size*2)
            em.add_assignment(Op(em, double_out, double_in, 'and', 'not',
                          'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
                          ))
            if (type(dout) == LogicVec):
                em.add_assignment(Op(em, dout, f'{double_out.getNameRead()}({size-1} downto 0) or ' + \
                    f'{double_out.getNameRead()}({2*size-1} downto {size})'))
            else:
                for i in range(0, size):
                    em.add_assignment(Op(em, (dout, i), f'{double_out.getNameRead(i)} or {double_out.getNameRead(i+size)}'))
    em.add_comment('Priority Masking End\n')
