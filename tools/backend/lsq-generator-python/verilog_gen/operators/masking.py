from verilog_gen.emitters import Emitter
from verilog_gen.utils import isPow2
from verilog_gen.ir import Op, Val


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

           dout1= 010000        dout2= 000100      dout3= 00001
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
    from verilog_gen.signals import LogicVecArray, LogicVec, LogicArray
    if isinstance(din, LogicVecArray):
        assert (reverse == False)
        for i in range(0, din.size):
            size = din.length
            double_in = LogicVec(em, em.get_temp(f'double_in_{i}'), 'w', size*2)
            for j in range(0, size):
                em.add_assignment((double_in, j), Val(din, j, i))
                em.add_assignment((double_in, j+size), Val(din, j, i))
            double_out = LogicVec(em, em.get_temp(f'double_out_{i}'), 'w', size*2)
            # TODO: Double check whether the brackets are correct
            em.add_assignment(double_out, double_in & ~(double_in - (Val(0, size).concat(base))))
            for j in range(0, size):
                em.add_assignment((dout, j, i), Val(double_out, j) | Val(double_out, j+size))
    else:
        if reverse:
            if isinstance(din, LogicArray):
                size = din.length
            else:
                size = din.size
            double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
            for i in range(0, size):
                em.add_assignment((double_in, i), Val(din, size-1-i))
                em.add_assignment((double_in, i+size), Val(din, size-1-i))
            base_rev = LogicVec(em, em.get_temp('base_rev'), 'w', size)
            for i in range(0, size):
                em.add_assignment((base_rev, i), Val(base, size-1-i))
            double_out = LogicVec(em, em.get_temp('double_out'), 'w', size*2)
            em.add_assignment(double_out, double_in & ~(double_in - (Val(0, size).concat(base_rev))))
            for i in range(0, size):
                em.add_assignment((dout, size-1-i), Val(double_out, i) | Val(double_out, i+size))
        else:
            if isinstance(din, LogicArray):
                size = din.length
                double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
                for i in range(0, size):
                    em.add_assignment((double_in, i), Val(din, i))
                    em.add_assignment((double_in, i+size), Val(din, i))
            else:
                size = din.size
                double_in = LogicVec(em, em.get_temp('double_in'), 'w', size*2)
                em.add_assignment(double_in, din & din)
            double_out = LogicVec(em, em.get_temp('double_out'), 'w', size*2)
            em.add_assignment(double_out, double_in & ~(double_in - (Val(0, size).concat(base))))
            if isinstance(dout, LogicVec):
                # TODO: Have indexing function
                em.add_assignment(Op(em, dout, f'{double_out.getNameRead()}({size-1} downto 0) or ' + \
                    f'{double_out.getNameRead()}({2*size-1} downto {size})'))
            else:
                for i in range(0, size):
                    em.add_assignment((dout, i), Val(double_out, i) | Val(double_out, i+size))
    em.add_comment('Priority Masking End\n')
