#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# This file contains all needed functions for the LSQ generation process (VHDL)
import math
import argparse
from configs import *
from utils import *

#===----------------------------------------------------------------------===#
# Global Parameter Initialization
#===----------------------------------------------------------------------===#

tabLevel = 0
tempCount = 0
signalInitString = ''
portInitString   = ''
regInitString    = ''
library          = 'library IEEE;\nuse IEEE.std_logic_1164.all;\nuse IEEE.numeric_std.all;\n\n'

#===----------------------------------------------------------------------===#
# VHDL Signal Definition
#===----------------------------------------------------------------------===#
# This section defined Python classes that generate VHDL signal declarations.
# 
# - class Logic         : (std_logic) one‑bit signal wire / port / register
# - class LogicVec      : (std_logic_vector) Multi-bit signal.
# - class LogicArray    : (Multiple std_logic) Array of individual std_logic signals.
# - class LogicVecArray : (Multiple std_logic_vector) Array of std_logic_vector signals.

#
# std_logic bit
#
class Logic:
    """
    A one-bit VHDL std_logic signal.

    Logic class encapsulates wires, ports, and registers in the code generator,
    handling name with '_i', '_o', '_q', '_d' suffixes.

    Attributes:
        name (str): The base name of the signal.
        type (str): 
            'i' input port      (<name>_i: in std_logic)
            'o' output port     (<name>_o: out std_logic)
            'w' internal wire   (signal <name>: std_logic)
            'r' register        (<name>_q) for the registered value
                                (<name>_d) for the next-cycled value
    
    Methods:
        getNameRead(): Returns the name we should use when reading the signal. (e.g. <name>_q for a register type)
        getNameWrite(): Returns the name to write to. (e.g. <name>_d for a register type)
        signalInit(): Appends the VHDL signal/port declaration.
        regInit(): Appends the VHDL register initialization block.
    """
        
    # Signal name
    name = ''
    # Signal type, 'i' for input, 'o' for output, 'w' for wire, 'r' for register
    type = ''
    def __init__(self, name: str, type: str = 'w', init: bool = True) -> None:
        """
        init: If True, immediately generates the corresponding std_logic in VHDL.
              True when we instantiate Logic.
              False when we instantiate LogicVec, LogicArray, and LogicVecArray.
        """
        # Type should be one of the four types.
        assert(type in ('i','o','w','r'))
        self.name = name
        self.type = type
        if (init):
            self.signalInit()
    def __repr__(self) -> str:
        """
        Print Logic with useful information.
        """
        # Signal type
        type = ''
        if (self.type   == 'w'):
            type = 'wire'
        elif(self.type == 'i'):
            type = 'input'
        elif(self.type == 'o'):
            type = 'output'
        elif(self.type  == 'r'):
            type = 'reg'
        return f'name: {self.name}\n' + f'type: {type}\n' + f'size: single bit\n'
    def getNameRead(self, sufix = '') -> str:
        """
        Returns the name we should use when reading the signal.

        Example (Pseudo-code)
            If you want to do "Logic a = Logic b + Logic c"
            -> getNameWrite(a) = getNameRead(b) + getNameRead(c)
        """
        if (self.type == 'w'):
            return self.name + sufix
        elif(self.type == 'r'):
            return self.name + sufix + '_q'
        elif(self.type == 'i'):
            return self.name + sufix + '_i'
        elif(self.type == 'o'):
            raise TypeError(f'Cannot read from the output signal \"{self.name}\"!')
    def getNameWrite(self, sufix = '') -> str:
        """
        Returns the name to write to. 
        
        Example in the getNameRead() method.
        """
        if (self.type == 'w'):
            return self.name + sufix
        elif (self.type == 'r'):
            return self.name + sufix + '_d'
        elif (self.type == 'i'):
            raise TypeError(f'Cannot write to the input signal \"{self.name}\"!')
        elif (self.type == 'o'):
            return self.name + sufix + '_o'
    def signalInit(self, sufix = '') -> None:
        """
        Appends the appropriate declaration or port line for this signal to a global buffer.
        """
        global signalInitString
        global portInitString
        if (self.type == 'w'):
            signalInitString += f'\tsignal {self.name + sufix} : std_logic;\n'
        elif(self.type == 'r'):
            signalInitString += f'\tsignal {self.name + sufix}_d : std_logic;\n'
            signalInitString += f'\tsignal {self.name + sufix}_q : std_logic;\n'
        elif(self.type == 'i'):
            portInitString += ';\n'
            portInitString += f'\t\t{self.name + sufix}_i : in std_logic'
        elif(self.type == 'o'):
            portInitString += ';\n'
            portInitString += f'\t\t{self.name + sufix}_o : out std_logic'
    def regInit(self, enable = None, init = None) -> None:
        """
        Generates a clocked process snippet that sets up the register's behavior.
        For example,

        if (rst = '1') then
            <name>_q <= '0';
        elsif (rising_edge(clk)) then
            <name>_q <= <name>_d;
        end if;
        """
        global regInitString
        assert (self.type == 'r')
        if (init != None):
            regInitString += '\t\tif (rst = \'1\') then\n'
            regInitString += f'\t\t\t{self.getNameRead()} <= {IntToBits(init)};\n'
            regInitString += '\t\telsif (rising_edge(clk)) then\n'
        else:
            regInitString += '\t\tif (rising_edge(clk)) then\n'
        if (enable != None):
            regInitString += f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n'
            regInitString += f'\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n'
            regInitString += '\t\t\tend if;\n'
        else:
            regInitString += f'\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n'
        regInitString += '\t\tend if;\n'

#
# std_logic_vec
#
class LogicVec(Logic):
    """
    Like 'class Logic', but for M-bit vectors.

    Inherits all methods and suffix rules of Logic in default.
    Additionally, it has additional features.

    Attributes:
        size (int): bit-width of vector (M)
    
    Methods:
        Indexable reads/writes of LogicVec components
        Access a certain i-th bit of LogicVec via getNameRead(i), getNameWrite(i)

        LogicVec (size=3)    : "101"
        LogicArray (length=3): [1,
                                0,
                                1]
        LogicVecArray (size=3, length=2): [101,
                                           010]
    """
    # Signal name
    name = ''
    # Signal type, 'i' for input, 'o' for output, 'w' for wire, 'r' for register
    type = ''
    size = 1
    def __init__(self, name: str, type: str = 'w', size: int = 1, init: bool = True) -> None:
        Logic.__init__(self, name, type, False)
        assert(size > 0)
        self.size = size
        if (init):
            self.signalInit()
    def __repr__(self) -> str:
        # Signal type
        type = ''
        if (self.type   == 'w'):
            type = 'wire'
        elif (self.type == 'i'):
            type = 'input'
        elif (self.type == 'o'):
            type = 'output'
        elif(self.type  == 'r'):
            type = 'reg'
        return f'name: {self.name}\n' + f'type: {type}\n' + f'size: {self.size}\n'
    def getNameRead(self, i = None, sufix = '') -> str:
        if (i == None):
            return Logic.getNameRead(self, sufix)
        else:
            assert(i < self.size)
            return Logic.getNameRead(self, sufix) + f'({i})'
    def getNameWrite(self, i = None, sufix = '') -> str:
        if (i == None):
            return Logic.getNameWrite(self, sufix)
        else:
            assert(i < self.size)
            return Logic.getNameWrite(self, sufix) + f'({i})'
    def signalInit(self, sufix = ''):
        global signalInitString
        global portInitString
        if (self.type == 'w'):
            signalInitString += f'\tsignal {self.name + sufix} : std_logic_vector({self.size-1} downto 0);\n'
        elif(self.type == 'r'):
            signalInitString += f'\tsignal {self.name + sufix}_d : std_logic_vector({self.size-1} downto 0);\n'
            signalInitString += f'\tsignal {self.name + sufix}_q : std_logic_vector({self.size-1} downto 0);\n'
        elif(self.type == 'i'):
            portInitString += ';\n'
            portInitString += f'\t\t{self.name + sufix}_i : in std_logic_vector({self.size-1} downto 0)'
        elif(self.type == 'o'):
            portInitString += ';\n'
            portInitString += f'\t\t{self.name + sufix}_o : out std_logic_vector({self.size-1} downto 0)'
    def regInit(self, enable = None, init = None) -> None:
        global regInitString
        assert (self.type == 'r')
        if (init != None):
            regInitString += '\t\tif (rst = \'1\') then\n'
            regInitString += f'\t\t\t{self.getNameRead()} <= {IntToBits(init, self.size)};\n'
            regInitString += '\t\telsif (rising_edge(clk)) then\n'
        else:
            regInitString += '\t\tif (rising_edge(clk)) then\n'
        if (enable != None):
            regInitString += f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n'
            regInitString += f'\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n'
            regInitString += '\t\t\tend if;\n'
        else:
            regInitString += f'\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n'
        regInitString += '\t\tend if;\n'

#
# An array of std_logic
#
class LogicArray(Logic):
    """
    Represents a N-length array of one-bit VHDL std_logic.
    Generates total of N one-bit std_logic.

    Each element (total N) is generated as a separate Logic(name + f'_{i}', type)
    For example,
        signal <name>_0 : std_logic;
        signal <name>_1 : std_logic;
        ...
        signal <name>_{N-1} : std_logic;
        
    Attributes:
        length (int): number of elements in the array.
    
    Methods:
        Indexable reads/writes of LogicArray components
        Access a certain i-th element of LogicArray via getNameRead(i), getNameWrite(i)
    """
    length = 1
    def __init__(self, name: str, type: str = 'w', length: int = 1):
        self.length = length
        Logic.__init__(self, name, type, False)
        self.signalInit()
    def __repr__(self) -> str:
        return Logic.__repr__(self) + f'array length: {self.length}'
    def getNameRead(self, i) -> str:
        assert i in range(0, self.length)
        return Logic.getNameRead(self, f'_{i}')
    def getNameWrite(self, i) -> str:
        assert i in range(0, self.length)
        return Logic.getNameWrite(self, f'_{i}')
    def signalInit(self) -> None:
        for i in range(0, self.length):
            Logic.signalInit(self, f'_{i}')
    def __getitem__(self, i) -> Logic:
        assert i in range(0, self.length)
        return Logic(self.name + f'_{i}', self.type, False)
    def regInit(self, enable = None, init = None) -> None:
        global regInitString
        assert (self.type == 'r')
        if (init != None):
            regInitString += '\t\tif (rst = \'1\') then\n'
            for i in range(0, self.length):
                regInitString += f'\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i])};\n'
            regInitString += '\t\telsif (rising_edge(clk)) then\n'
        else:
            regInitString += '\t\tif (rising_edge(clk)) then\n'
        if (enable != None):
            for i in range(0, self.length):
                regInitString += f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n'
                regInitString += f'\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n'
                regInitString += '\t\t\tend if;\n'
        else:
            for i in range(0, self.length):
                regInitString += f'\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n'
        regInitString += '\t\tend if;\n'

#
# An array of std_logic vector
#
class LogicVecArray(LogicVec):
    """
    Represents a N-length array of M-bit VHDL std_logic_vec.
    Generates total of N M-bit std_logic_vec.

    Each element (total N) is generated as a separate LogicVec
    For example,
        signal <name>_0 : std_logic_vector(M-1 downto 0);
        signal <name>_1 : std_logic_vector(M-1 downto 0);
        …
        signal <name>_{N-1} : std_logic_vector(M-1 downto 0);

    Attributes:
        length (int): number of entries (N).
        size   (int): bit-width of each vector (M).
    
    Methods:
        Indexable reads/writes of LogicVecArray components
        Access a certain i-th LogicVec of LogicVecArray via getNameRead(i), getNameWrite(i)
    """
    length = 1
    def __init__(self, name: str, type: str = 'w', length: int = 1, size: int = 1):
        self.length = length
        LogicVec.__init__(self, name, type, size, False)
        self.signalInit()
    def __repr__(self) -> str:
        return LogicVec.__repr__(self) + f'array length: {self.length}'
    def getNameRead(self, i, j = None) -> str:
        assert i in range(0, self.length)
        return LogicVec.getNameRead(self, j, f'_{i}')
    def getNameWrite(self, i, j = None) -> str:
        assert i in range(0, self.length)
        return LogicVec.getNameWrite(self, j, f'_{i}')
    def signalInit(self) -> None:
        for i in range(0, self.length):
            LogicVec.signalInit(self, f'_{i}')
    def __getitem__(self, i) -> LogicVec:
        assert i in range(0, self.length)
        return LogicVec(self.name + f'_{i}', self.type, self.size, False)
    def regInit(self, enable = None, init = None) -> None:
        global regInitString
        assert (self.type == 'r')
        if (init != None):
            regInitString += '\t\tif (rst = \'1\') then\n'
            for i in range(0, self.length):
                regInitString += f'\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i], self.size)};\n'
            regInitString += '\t\telsif (rising_edge(clk)) then\n'
        else:
            regInitString += '\t\tif (rising_edge(clk)) then\n'
        if (enable != None):
            for i in range(0, self.length):
                regInitString += f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n'
                regInitString += f'\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n'
                regInitString += '\t\t\tend if;\n'
        else:
            for i in range(0, self.length):
                regInitString += f'\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n'
        regInitString += '\t\tend if;\n'

#===----------------------------------------------------------------------===#
# Unit Operator
#===----------------------------------------------------------------------===#

def Op(out, *list_in) -> str:    
    """
    Generates a proper VHDL assignment statement.

    Args:
        out: LHS of the assignment
        *list_in: A sequence of RHS elements

    Example: Op(valid, a,'when', b, 'else', 0)
            valid <= a when b else '0'
    
    """
    global tabLevel
    if type(out) == tuple:
        if len(out) == 2:
            str_ret = '\t'*tabLevel + f'{out[0].getNameWrite(out[1])} <='
        else:
            str_ret = '\t'*tabLevel + f'{out[0].getNameWrite(out[1], out[2])} <='
    else:
        str_ret = '\t'*tabLevel + f'{out.getNameWrite()} <='
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

#===----------------------------------------------------------------------===#
# Helper Functions
#===----------------------------------------------------------------------===#

def getTemp(name) -> str:
    global tempCount
    return f'TEMP_{tempCount}_{name}'

def useTemp() -> None:
    global tempCount
    tempCount += 1

def isPow2(value: int) -> bool:
    return (value & (value-1) == 0) and value != 0

def log2Ceil(value: int) -> int:
    return math.ceil(math.log2(value))

def WrapAdd(out, in_a, in_b, max: int) -> str:
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

    global tabLevel
    str_ret = '\t'*tabLevel + '-- WrapAdd Begin\n'
    str_ret += '\t'*tabLevel + f'-- WrapAdd({out.name}, {in_a.name}, {in_b.name}, {max})\n'
    if (isPow2(max)):
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + unsigned({in_b.getNameRead()}));\n'
    else:
        useTemp()
        sum = LogicVec(getTemp('sum'), 'w', out.size + 1)
        res = LogicVec(getTemp('res'), 'w', out.size + 1)
        str_ret += '\t'*tabLevel + f'{sum.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned(\'0\' & {in_a.getNameRead()}) + unsigned(\'0\' & {in_b.getNameRead()}));\n'
        str_ret += '\t'*tabLevel + f'{res.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({sum.getNameRead()}) - {max}) ' + \
            f'when {sum.getNameRead()} >= {max} else {sum.getNameRead()};\n'
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= {res.getNameRead()}({out.size-1} downto 0);\n'
    str_ret += '\t'*tabLevel + '-- WrapAdd End\n\n'
    return str_ret

def WrapAddConst(out, in_a, const: int, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a + const
    else:
        if in_a + const >= max:
            out = in_a + const - max
        else:
            out = in_a + const
    """

    global tabLevel
    str_ret = '\t'*tabLevel + '-- WrapAdd Begin\n'
    str_ret += '\t'*tabLevel + f'-- WrapAdd({out.name}, {in_a.name}, {const}, {max})\n'
    if (isPow2(max)):
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + {const});\n'
    else:
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - {max - const}) ' + \
            f'when {in_a.getNameRead()} >= {max - const} else ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) + {const}));\n'
    str_ret += '\t'*tabLevel + '-- WrapAdd End\n\n'
    return str_ret

def WrapSub(out, in_a, in_b, max: int) -> str:
    """
    if "max" is power of 2:
        out = in_a - in_b
    else:
        if in_a >= in_b:
            out = in_a - in_b
        else:
            out = (in_a + max) - in_b
    """

    global tabLevel
    str_ret = '\t'*tabLevel + '-- WrapSub Begin\n'
    str_ret += '\t'*tabLevel + f'-- WrapSub({out.name}, {in_a.name}, {in_b.name}, {max})\n'
    if (isPow2(max)):
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - unsigned({in_b.getNameRead()}));\n'
    else:
        str_ret += '\t'*tabLevel + f'{out.getNameWrite()} <= ' + \
            f'std_logic_vector(unsigned({in_a.getNameRead()}) - unsigned({in_b.getNameRead()})) ' + \
            f'when {in_a.getNameRead()} >= {in_b.getNameRead()} else\n' + '\t'*(tabLevel+1) + \
            f'std_logic_vector({max} - unsigned({in_b.getNameRead()}) + unsigned({in_a.getNameRead()}));\n'
    str_ret += '\t'*tabLevel + '-- WrapAdd End\n\n'
    return str_ret

#===----------------------------------------------------------------------===#
# Cyclic Left Shift
#===----------------------------------------------------------------------===#
# The following functions implement cyclic left shifts:
#   RotateLogicVec()      : Recursively shift a single vector.
#   RotateLogicArray()    : Recursively shift an array of single-bit elements.
#   RotateLogicVecArray() : Recursively shift an array of vectors.
#   -> These are called only internally by CyclicLeftShift().
#
# CyclicLeftShift():
#   Detects the type of `din` and dispatches to the appropriate implementation.

def RotateLogicVec(dout, din, distance, layer) -> str:
    """
    Recursively perform a cyclic left shift of the vector "din" by the amount 
    specified in "distance".

    Parameters:
        dout     (LogicVec): Destination vector to hold the shifted result.
        din      (LogicVec): Source vector to be shifted.
        distance (LogicVec): Binary vector representing the shift amount.
        layer    (int)     : Current recursion layer; set to "distance.size-1" when called initially.
    
        The "layer" parameter is used internally to control recursion depth and
        should always start at "distance.size - 1".

    Returns:
        str_ret (str): A VHDL code snippet implementing the cyclic left shift.

    Usage:
        (Called only internally by CyclicLeftShift)
        RotateLogicVec(dout, din, distance, distance.size - 1)

        When this method is called, "layer" is always "distance.size - 1".
        "layer" is just for an recursive action.
        

    Example: 
        Input:  din  = "01110010", distance = 3
        Output: dout = "10010011"
    """

    global tabLevel
    str_ret = ''
    length = din.size
    if (layer == 0):
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        useTemp()
        res = LogicVec(getTemp('res'), 'w', length)
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += RotateLogicVec(dout, res, distance, layer-1)
    return str_ret

def RotateLogicArray(dout, din, distance, layer) -> str:
    """
    Recursively perform a cyclic left shift of LogicArray "din" by the amount 
    specified in "distance".
    
    Identical in behavior to RotateLogicVec, but operates on multiple VHDL single-bit std_logic
    instead of std_logic_vector.
    
    """

    global tabLevel
    str_ret = ''
    length = din.length
    if (layer == 0):
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        useTemp()
        res = LogicArray(getTemp('res'), 'w', length)
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += RotateLogicArray(dout, res, distance, layer-1)
    return str_ret

def RotateLogicVecArray(dout, din, distance, layer) -> str:    
    """
    Recursively perform a cyclic left shift of the LogicVecArray "din" by the amount 
    specified in "distance".
    
    Identical in behavior to RotateLogicVec, but operates on multiple VHDL vectors std_logic_vector.
    For every LogicVec in LogicVecArray, cyclic left shift by "distance".
    
    Example:
        din = "11001001
               11100011"
        distance = 2
        
        -> dout = "00100111     (Cyclic Left Shift of each vector by 2)
                   10001111"
    """

    global tabLevel
    str_ret = ''
    length = din.length
    if (layer == 0):
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
    else:
        useTemp()
        res = LogicVecArray(getTemp('res'), 'w', length, dout.size)
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{res.getNameWrite(i)} <= {din.getNameRead((i-2**layer)%length)} ' + \
                f'when {distance.getNameRead(layer)} else {din.getNameRead(i)};\n'
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += RotateLogicVecArray(dout, res, distance, layer-1)
    return str_ret

def CyclicLeftShift(dout, din, distance) -> str:
    """
    Execute a cyclic left shift operation based on the type of "din"

    This function wraps the three implementations:
        - RotateLogicVec        : when "din" is LogicVec
        - RotateLogicArray      : when "din" is LogicArray
        - RotateLogicVecArray   : when "din" is LogicVecArray

    Parameters:
        dout    : Destination signal to receive the shifted data.
        din     : Source data to be shifted.
        distance: Binary vector specifying how many positions to shift.

    Returns:
        str_ret : A VHDL code snippet (with indentation) implementing the cyclic left shift.
    """

    global tabLevel
    str_ret = '\t'*tabLevel + '-- Shifter Begin\n'
    str_ret += '\t'*tabLevel + f'-- CyclicLeftShift({dout.name}, {din.name}, {distance.name})\n'
    if (type(din) == LogicArray):
        str_ret += RotateLogicArray(dout, din, distance, distance.size-1)
    elif (type(din) == LogicVecArray):
        str_ret += RotateLogicVecArray(dout, din, distance, distance.size-1)
    else:
        str_ret += RotateLogicVec(dout, din, distance, distance.size-1)
    str_ret += '\t'*tabLevel + '-- Shifter End\n\n'
    return str_ret


#===----------------------------------------------------------------------===#
# Reduction
#===----------------------------------------------------------------------===#
# The following functions implement cyclic left shifts:
#   ReduceLogicVec()      : Recursively reduce a single vector.
#   ReduceLogicArray()    : Recursively reduce an array of single-bit elements.
#   ReduceLogicVecArray() : Recursively reduce an array of vectors.
#   -> These are called only internally by Reduce().
#
# Reduce():
#   Detects the type of `din` and dispatches to the appropriate implementation.

def ReduceLogicVec(dout, din, operator, length) -> str:
    """
    Recursively reduce the vector "din" by "operator".

    Parameters:
        dout     (Logic)   : Destination std_logic to hold the reduced result.
        din      (LogicVec): Source vector to be reduced.
        operator (str)     : 'and', 'or', ...
        length   (int)     : Current recursion length;
                             set to "2**(log2Ceil(din.size) - 1)" when called initially.
    
        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

    Returns:
        str_ret (str): A VHDL code snippet implementing the LogicVec reduction.

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
    global tabLevel
    str_ret = ''
    if (length == 1):
        str_ret += '\t'*tabLevel + f'{dout.getNameWrite()} <= ' + \
            f'{din.getNameRead(0)} {operator} {din.getNameRead(1)};\n'
    else:
        useTemp()
        res = LogicVec(getTemp('res'), 'w', length)
        for i in range(0, din.size - length):
            str_ret += '\t'*tabLevel + f'{res.getNameWrite(i)} <= ' + \
            f'{din.getNameRead(i)} {operator} {din.getNameRead(i+length)};\n'
        for i in range(din.size - length, length):
            str_ret += '\t'*tabLevel + f'{res.getNameWrite(i)} <= ' + \
            f'{din.getNameRead(i)};\n'
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += ReduceLogicVec(dout, res, operator, length//2)
    return str_ret

def ReduceLogicArray(dout, din, operator, length) -> str:
    """
    Recursively perform reduction of LogicArray "din" by "operator".
    
    Identical in behavior to ReduceLogicVec, but operates on multiple VHDL single-bit std_logic
    instead of std_logic_vector.
    """

    global tabLevel
    str_ret = ''
    if (length == 1):
        str_ret += Op(dout, din[0], operator, din[1])
    else:
        useTemp()
        res = LogicArray(getTemp('res'), 'w', length)
        for i in range(0, din.length - length):
            str_ret += Op(res[i], din[i], operator, din[i+length])
        for i in range(din.length - length, length):
            str_ret += Op(res[i], din[i])
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += ReduceLogicArray(dout, res, operator, length//2)
    return str_ret

def ReduceLogicVecArray(dout, din, operator, length) -> str:
    """
    Recursively perform reduction of the LogicVecArray "din" by "operator".
    
    Parameters:
        dout     (LogicVec)     : Destination std_logic_vector to hold the reduced result.
        din      (LogicVecArray): Source LogicVecArray to be reduced.
        operator (str)          : 'and', 'or', ...
        length   (int)          : Current recursion length;
                                  set to "2**(log2Ceil(din.size) - 1)" when called initially.
    
        The "length" parameter is used internally to control recursion depth and
        should always start at "2**(log2Ceil(din.size) - 1)".

    Returns:
        str_ret (str): A VHDL code snippet implementing the LogicVecArray reduction.

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
    global tabLevel
    str_ret = ''
    if (length == 1):
        str_ret += Op(dout, din[0], operator, din[1])
    else:
        useTemp()
        res = LogicVecArray(getTemp('res'), 'w', length, dout.size)
        for i in range(0, din.length - length):
            str_ret += Op(res[i], din[i], operator, din[i+length])
        for i in range(din.length - length, length):
            str_ret += Op(res[i], din[i])
        str_ret += '\t'*tabLevel + '-- Layer End\n'
        str_ret += ReduceLogicVecArray(dout, res, operator, length//2)
    return str_ret

def Reduce(dout, din, operator, comment: bool = True) -> str:
    """
    Execute reduction based on the type of "din"

    This function wraps the three implementations:
        - ReduceLogicVec        : when "din" is LogicVec
        - ReduceLogicArray      : when "din" is LogicArray
        - ReduceLogicVecArray   : when "din" is LogicVecArray

    Parameters:
        dout    : Destination signal to receive the reduced data.
        din     : Source data to be reduced.
        operator: types of operator for the reduction
        comment : Turn on/off adding VHDL comment lines.

    Returns:
        str_ret : A VHDL code snippet (with indentation) implementing the reduction.
    """

    global tabLevel
    str_ret = ''
    if (comment):
        str_ret += '\t'*tabLevel + '-- Reduction Begin\n'
        str_ret += '\t'*tabLevel + f'-- Reduce({dout.name}, {din.name}, {operator})\n'
    if (type(din) == LogicVec):
        if (din.size == 1):
            str_ret += Op(dout, (din, 0))
        else:
            length = 2**(log2Ceil(din.size) - 1)
            str_ret += ReduceLogicVec(dout, din, operator, length)
    else:
        if (din.length == 1):
            str_ret += Op(dout, din[0])
        else:
            length = 2**(log2Ceil(din.length) - 1)
            if (type(din) == LogicArray):
                str_ret += ReduceLogicArray(dout, din, operator, length)
            else:
                str_ret += ReduceLogicVecArray(dout, din, operator, length)
    if (comment):
        str_ret += '\t'*tabLevel + '-- Reduction End\n\n'
    return str_ret


#===----------------------------------------------------------------------===#
# Multiplexer
#===----------------------------------------------------------------------===#
# Mux1H    : One-hot select elements of `din` using `sel`
# Mux1HROM : Special multiplexer for the Group Allocator ROM.
# MuxIndex : 
# MuxLookUp: 

def Mux1H(dout, din, sel, j = None) -> str:
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
    global tabLevel
    str_ret = '\t'*tabLevel + '-- Mux1H Begin\n'
    str_ret += '\t'*tabLevel + f'-- Mux1H({dout.name}, {din.name}, {sel.name})\n'
    useTemp()

    # din is always LogicVecArray
    if (type(din) == LogicVecArray):
        length = din.length
        size = din.size
        mux = LogicVecArray(getTemp('mux'), 'w', length, din.size)
    elif (type(din) == LogicArray):
        length = din.length
        size = None
        mux = LogicArray(getTemp('mux'), 'w', length)
    else:
        length = din.size
        size = None
        mux = LogicArray(getTemp('mux'), 'w', length)

    str_zero = Zero(size)
    if (j == None):
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{mux.getNameWrite(i)} <= {din.getNameRead(i)} ' +\
                f'when {sel.getNameRead(i)} = \'1\' else {str_zero};\n'
    else:
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'{mux.getNameWrite(i)} <= {din.getNameRead(i)} ' +\
                f'when {sel.getNameRead(i, j)} = \'1\' else {str_zero};\n'

    str_ret += Reduce(dout, mux, 'or', False)
    str_ret += '\t'*tabLevel + '-- Mux1H End\n\n'
    return str_ret

def Mux1HROM(dout, din, sel, func = IntToBits) -> str:
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

    global tabLevel
    str_ret = '\t'*tabLevel + '-- Mux1H For Rom Begin\n'
    str_ret += '\t'*tabLevel + f'-- Mux1H({dout.name}, {sel.name})\n'
    useTemp()
    mlen   = sel.length
    size   = dout.size
    str_zero = Zero(size)
    if (type(dout) == LogicVecArray):
        length = dout.length
        for i in range(0, length):
            str_ret += '\t'*tabLevel + f'-- Loop {i}\n'
            mux = LogicVecArray(getTemp(f'mux_{i}'), 'w', mlen, size)
            for j in range(0, mlen):
                str_value = func(GetValue(din[j], i), size)
                if (str_value == str_zero):
                    str_ret += '\t'*tabLevel + f'{mux.getNameWrite(j)} <= {str_zero};\n'
                else:
                    str_ret += '\t'*tabLevel + f'{mux.getNameWrite(j)} <= {str_value} ' + \
                        f'when {sel.getNameRead(j)} else {str_zero};\n'
            str_ret += Reduce(dout[i], mux, 'or', False)
    else:   # type(dout) == LogicVec
        mux = LogicVecArray(getTemp(f'mux'), 'w', mlen, size)
        for j in range(0, mlen):
            str_value = func(din[j], size)
            if (str_value == str_zero):
                str_ret += '\t'*tabLevel + f'{mux.getNameWrite(j)} <= {str_zero};\n'
            else:
                str_ret += '\t'*tabLevel + f'{mux.getNameWrite(j)} <= {str_value} ' + \
                    f'when {sel.getNameRead(j)} else {str_zero};\n'
        str_ret += Reduce(dout, mux, 'or', False)
    str_ret += '\t'*tabLevel + '-- Mux1H For Rom End\n\n'
    return str_ret

def MuxIndex(din, sel) -> str:
    """
    Generate a VHDL array-index expression for selecting an element
    """
    return f'{din.getNameRead()}(to_integer(unsigned({sel.getNameRead()})))'

def MuxLookUp(dout, din, sel) -> str:
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
    global tabLevel
    str_ret = '\t'*tabLevel + '-- MuxLookUp Begin\n'
    str_ret += '\t'*tabLevel + f'-- MuxLookUp({dout.name}, {din.name}, {sel.name})\n'


    length = din.length
    size   = sel.size
    str_ret += '\t'*tabLevel + f'{dout.getNameWrite()} <= \n'

    for i in range(0, length):
        str_ret += '\t'*tabLevel + f'{din.getNameRead(i)} ' +\
            f'when ({sel.getNameRead()} = {IntToBits(i, size)}) else\n'
    if (type(dout) == LogicVec):
        str_ret += '\t'*tabLevel + f'{Zero(dout.size)};\n'
    else:
        str_ret += '\t'*tabLevel + f'\'0\';\n'

    str_ret += '\t'*tabLevel + '-- MuxLookUp End\n\n'
    return str_ret

def VecToArray(dout, din) -> str:
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
        str_ret += Op((dout, i), (din, i))
    return str_ret

def CyclicPriorityMasking(dout, din, base, reverse = False) -> str:
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

    global tabLevel
    str_ret = '\t'*tabLevel + '-- Priority Masking Begin\n'
    str_ret += '\t'*tabLevel + f'-- CyclicPriorityMask({dout.name}, {din.name}, {base.name})\n'
    useTemp()
    if (type(din) == LogicVecArray):
        assert(reverse == False)
        for i in range(0, din.size):
            size = din.length
            double_in = LogicVec(getTemp(f'double_in_{i}'), 'w', size*2)
            for j in range(0, size):
                str_ret += Op((double_in, j), (din, j, i))
                str_ret += Op((double_in, j+size), (din, j, i))
            double_out = LogicVec(getTemp(f'double_out_{i}'), 'w', size*2)
            str_ret += Op(double_out, double_in, 'and', 'not',
                'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
            )
            for j in range(0, size):
                str_ret += '\t'*tabLevel + f'{dout.getNameWrite(j, i)} <= ' + \
                    f'{double_out.getNameRead(j)} or {double_out.getNameRead(j+size)};\n'
    else:
        if reverse:
            if (type(din) == LogicArray):
                size = din.length
            else:
                size = din.size
            double_in = LogicVec(getTemp('double_in'), 'w', size*2)
            for i in range(0, size):
                str_ret += Op((double_in, i), (din, size-1-i))
                str_ret += Op((double_in, i+size), (din, size-1-i))
            base_rev   = LogicVec(getTemp('base_rev'), 'w', size)
            for i in range(0, size):
                str_ret += Op((base_rev, i), (base, size-1-i))
            double_out = LogicVec(getTemp('double_out'), 'w', size*2)
            str_ret += Op(double_out, double_in, 'and', 'not',
                'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base_rev, ')', ')'
            )
            for i in range(0, size):
                str_ret += '\t'*tabLevel + f'{dout.getNameWrite(size-1-i)} <= ' + \
                    f'{double_out.getNameRead(i)} or {double_out.getNameRead(i+size)};\n'
        else:
            if (type(din) == LogicArray):
                size = din.length
                double_in = LogicVec(getTemp('double_in'), 'w', size*2)
                for i in range(0, size):
                        str_ret += Op((double_in, i), (din, i))
                        str_ret += Op((double_in, i+size), (din, i))
            else:
                size = din.size
                double_in = LogicVec(getTemp('double_in'), 'w', size*2)
                str_ret += Op(double_in, din, '&', din)
            double_out = LogicVec(getTemp('double_out'), 'w', size*2)
            str_ret += Op(double_out, double_in, 'and', 'not',
                'std_logic_vector(', 'unsigned(', double_in, ')', '-', 'unsigned(', (0, size), '&', base, ')', ')'
            )
            if (type(dout) == LogicVec):
                str_ret += '\t'*tabLevel + f'{dout.getNameWrite()} <= ' + \
                    f'{double_out.getNameRead()}({size-1} downto 0) or ' + \
                    f'{double_out.getNameRead()}({2*size-1} downto {size});\n'
            else:
                for i in range(0, size):
                    str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= ' + \
                        f'{double_out.getNameRead(i)} or {double_out.getNameRead(i+size)};\n'
    str_ret += '\t'*tabLevel + '-- Priority Masking End\n\n'
    return str_ret

def BitsToOH(dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.

    Example:
        din  = "01"
        dout = "0010"
    """
    global tabLevel
    str_ret = '\t'*tabLevel + '-- Bits To One-Hot Begin\n'
    str_ret += '\t'*tabLevel + f'-- BitsToOH({dout.name}, {din.name})\n'
    for i in range(0, dout.size):
        str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= ' \
            f'\'1\' when {din.getNameRead()} = {IntToBits(i, din.size)} else \'0\';\n'
    str_ret += '\t'*tabLevel + '-- Bits To One-Hot End\n\n'
    return str_ret

def BitsToOHSub1(dout, din) -> str:
    """
    Convert a binary vector into its one-hot representation in VHDL.
    The result one-hot representation should be cyclic right shifted.

    Example:
        din  = "01"
        dout = "0001"
    """
    global tabLevel
    str_ret = '\t'*tabLevel + '-- Bits To One-Hot Begin\n'
    str_ret += '\t'*tabLevel + f'-- BitsToOHSub1({dout.name}, {din.name})\n'
    for i in range(0, dout.size):
        str_ret += '\t'*tabLevel + f'{dout.getNameWrite(i)} <= ' \
            f'\'1\' when {din.getNameRead()} = {IntToBits((i+1) % dout.size, din.size)} else \'0\';\n'
    str_ret += '\t'*tabLevel + '-- Bits To One-Hot End\n\n'
    return str_ret

def OHToBits(dout, din) -> str:
    """
    Generate VHDL code to convert a one-hot vector into its binary index.

    Example:
        din  = "0010"
        dout = "01"
    """
    global tabLevel
    str_ret = '\t'*tabLevel + '-- One-Hot To Bits Begin\n'
    str_ret += '\t'*tabLevel + f'-- OHToBits({dout.name}, {din.name})\n'
    size    = dout.size
    size_in = din.size
    useTemp()
    for i in range(0, size):
        temp_in  = LogicArray(getTemp(f'in_{i}'), 'w', size_in)
        temp_out = Logic(getTemp(f'out_{i}'), 'w')
        for j in range(0, size_in):
            if ((j // (2**i)) % 2 == 1):
                str_ret += Op((temp_in, j), (din, j))
            else:
                str_ret += Op((temp_in, j), '\'0\'')
        str_ret += Reduce(temp_out, temp_in, 'or', False)
        str_ret += Op((dout, i), temp_out)
    str_ret += '\t'*tabLevel + '-- One-Hot To Bits End\n\n'
    return str_ret

#===----------------------------------------------------------------------===#
# Module Generator Function Definitions
#===----------------------------------------------------------------------===#

def PortToQueueDispatcher(
    path_rtl:           str,
    name:               str,
    suffix:             str,
    numPorts:           int,
    numEntries:         int,
    bitsW:              int,
    portAddrW:          int
) -> str:
    # Initialize the global parameters
    global tabLevel
    global tempCount
    global signalInitString
    global portInitString
    global library
    tabLevel = 1
    tempCount = 0
    signalInitString = ''
    portInitString   = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    arch             = ''

    # IOs
    port_bits_i        = LogicVecArray('port_bits', 'i', numPorts, bitsW)
    port_valid_i       = LogicArray('port_valid', 'i', numPorts)
    port_ready_o       = LogicArray('port_ready', 'o', numPorts)
    entry_valid_i      = LogicArray('entry_valid', 'i', numEntries)
    entry_bits_valid_i = LogicArray('entry_bits_valid', 'i', numEntries)
    if (numPorts != 1):
        entry_port_idx_i   = LogicVecArray('entry_port_idx', 'i', numEntries, portAddrW)
    entry_bits_o       = LogicVecArray('entry_bits', 'o', numEntries, bitsW)
    entry_wen_o        = LogicArray('entry_wen', 'o', numEntries)
    queue_head_oh_i    = LogicVec('queue_head_oh', 'i', numEntries)

    # one-hot port index
    entry_port_valid = LogicVecArray('entry_port_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        if (numPorts == 1):
            arch += Op(entry_port_valid[i], 1)
        else:
            arch += BitsToOH(entry_port_valid[i], entry_port_idx_i[i])
    
    # Mux for the data/addr
    for i in range(0, numEntries):
        arch += Mux1H(entry_bits_o[i], port_bits_i, entry_port_valid[i])

    # Entries that request data/address from a any port
    entry_request_valid = LogicArray('entry_request_valid', 'w', numEntries)
    for i in range(0, numEntries):
        arch += Op(entry_request_valid[i], entry_valid_i[i], 'and', 'not', entry_bits_valid_i[i])

    # Entry-port pairs that the entry request the data/address from the port
    entry_port_request = LogicVecArray('entry_port_request', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(entry_port_request[i], entry_port_valid[i], 'when', entry_request_valid[i], 'else', 0)

    # Reduce the matrix for each entry to get the ready signal:
    # If one or more entries is requesting data/address from a certain port, ready is set high.
    port_ready_vec = LogicVec('port_ready_vec', 'w', numPorts)
    arch += Reduce(port_ready_vec, entry_port_request, 'or')
    arch += VecToArray(port_ready_o, port_ready_vec)

    # AND the request signal with valid, it shows entry-port pairs that are both valid and ready.
    entry_port_and = LogicVecArray('entry_port_and', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        for j in range(0, numPorts):
            arch += '\t'*tabLevel + f'{entry_port_and.getNameWrite(i, j)} <= ' \
                f'{entry_port_request.getNameRead(i, j)} and {port_valid_i.getNameRead(j)};\n'

    # For each port, the oldest entry recieves bit this cycle. The priority masking per port(column)
    # generates entry-port pairs that will tranfer data/address this cycle.
    entry_port_hs = LogicVecArray('entry_port_hs', 'w', numEntries, numPorts)
    arch += CyclicPriorityMasking(entry_port_hs, entry_port_and, queue_head_oh_i)
    
    # Reduce for each entry(row), which generates write enable signal for entries
    for i in range(0, numEntries):
        arch += Reduce(entry_wen_o[i], entry_port_hs[i], 'or')

    ######   Write To File  ######
    portInitString += '\n\t);'

    # Write to the file
    with open(f'{path_rtl}/{name}_core.vhd', 'a') as file:
        file.write('\n\n')
        file.write(library)
        file.write(f'entity {name + suffix} is\n')
        file.write(portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name + suffix} is\n')
        file.write(signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write('end architecture;\n')
    return

def QueueToPortDispatcher(
    path_rtl:           str,
    name:               str,
    suffix:             str,
    numPorts:           int,
    numEntries:         int,
    bitsW:              int,
    portAddrW:          int
) -> str:
    # Initialize the global parameters
    global tabLevel
    global tempCount
    global signalInitString
    global portInitString
    global library
    tabLevel = 1
    tempCount = 0
    signalInitString = ''
    portInitString   = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    arch             = ''

    # IOs
    if (bitsW != 0):
        port_bits_o    = LogicVecArray('port_bits', 'o', numPorts, bitsW)
    port_valid_o       = LogicArray('port_valid', 'o', numPorts)
    port_ready_i       = LogicArray('port_ready', 'i', numPorts)
    entry_valid_i      = LogicArray('entry_valid', 'i', numEntries)
    entry_bits_valid_i = LogicArray('entry_bits_valid', 'i', numEntries)
    if (numPorts != 1):
        entry_port_idx_i = LogicVecArray('entry_port_idx', 'i', numEntries, portAddrW)
    if (bitsW != 0):
        entry_bits_i   = LogicVecArray('entry_bits', 'i', numEntries, bitsW)
    entry_reset_o      = LogicArray('entry_reset', 'o', numEntries)
    queue_head_oh_i    = LogicVec('queue_head_oh', 'i', numEntries)

    # one-hot port index
    entry_port_valid = LogicVecArray('entry_port_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        if (numPorts == 1):
            arch += Op(entry_port_valid[i], 1)
        else:
            arch += BitsToOH(entry_port_valid[i], entry_port_idx_i[i])

    # This matrix shows entry-port pairs that the entry is linked with the port
    entry_port_request = LogicVecArray('entry_port_request', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(entry_port_request[i], entry_port_valid[i], 'when', entry_valid_i[i], 'else', 0)

    # For each port, the oldest entry send bits this cycle. The priority masking per port(column)
    # generates entry-port pairs that will tranfer data/address this cycle.
    # It is also used as one-hot select signal for data Mux.
    entry_port_request_prio = LogicVecArray('entry_port_request_prio', 'w', numEntries, numPorts)
    arch += CyclicPriorityMasking(entry_port_request_prio, entry_port_request, queue_head_oh_i)

    if (bitsW != 0):
        for j in range(0, numPorts):
            arch += Mux1H(port_bits_o[j], entry_bits_i, entry_port_request_prio, j)

    # Mask the matrix with dataValid
    entry_port_request_valid = LogicVecArray('entry_port_request_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(entry_port_request_valid[i], entry_port_request_prio[i],
        'when', entry_bits_valid_i[i], 'else', 0)
    
    # Reduce the matrix for each port to get the valid signal:
    # If an entry is providing data/address from a certain port, valid is set high.
    port_valid_vec = LogicVec('port_valid_vec', 'w', numPorts)
    arch += Reduce(port_valid_vec, entry_port_request_valid, 'or')
    arch += VecToArray(port_valid_o, port_valid_vec)

    # AND the request signal with ready, it shows entry-port pairs that are both valid and ready.
    entry_port_hs = LogicVecArray('entry_port_hs', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        for j in range(0, numPorts):
            arch += '\t'*tabLevel + f'{entry_port_hs.getNameWrite(i, j)} <= ' \
                f'{entry_port_request_valid.getNameRead(i, j)} and {port_ready_i.getNameRead(j)};\n'

    # Reduce for each entry(row), which generates reset signal for entries
    for i in range(0, numEntries):
        arch += Reduce(entry_reset_o[i], entry_port_hs[i], 'or')

    ######   Write To File  ######
    portInitString += '\n\t);'

    # Write to the file
    with open(f'{path_rtl}/{name}_core.vhd', 'a') as file:
        file.write('\n\n')
        file.write(library)
        file.write(f'entity {name + suffix} is\n')
        file.write(portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name + suffix} is\n')
        file.write(signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write('end architecture;\n')
    return

def GroupAllocator(path_rtl: str, name: str, suffix: str, configs: Configs) -> str:
    # Initialize the global parameters
    global tabLevel
    global tempCount
    global signalInitString
    global portInitString
    global regInitString
    global library
    tabLevel = 1
    tempCount = 0
    signalInitString = ''
    portInitString   = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    regInitString    = '\tprocess (clk, rst) is\n' + '\tbegin\n'
    arch             = ''

    # IOs
    group_init_valid_i = LogicArray('group_init_valid', 'i', configs.numGroups)
    group_init_ready_o = LogicArray('group_init_ready', 'o', configs.numGroups)

    ldq_tail_i         = LogicVec('ldq_tail', 'i', configs.ldqAddrW)
    ldq_head_i         = LogicVec('ldq_head', 'i', configs.ldqAddrW)
    ldq_empty_i        = Logic('ldq_empty', 'i')

    stq_tail_i         = LogicVec('stq_tail', 'i', configs.stqAddrW)
    stq_head_i         = LogicVec('stq_head', 'i', configs.stqAddrW)
    stq_empty_i        = Logic('stq_empty', 'i')

    ldq_wen_o          = LogicArray('ldq_wen', 'o', configs.numLdqEntries)
    num_loads_o        = LogicVec('num_loads', 'o', configs.emptyLdAddrW)
    num_loads          = LogicVec('num_loads', 'w', configs.emptyLdAddrW)
    if (configs.ldpAddrW > 0):
        ldq_port_idx_o = LogicVecArray('ldq_port_idx', 'o', configs.numLdqEntries, configs.ldpAddrW)

    stq_wen_o          = LogicArray('stq_wen', 'o', configs.numStqEntries)
    num_stores_o       = LogicVec('num_stores', 'o', configs.emptyStAddrW)
    num_stores         = LogicVec('num_stores', 'w', configs.emptyStAddrW)
    if (configs.stpAddrW > 0):
        stq_port_idx_o = LogicVecArray('stq_port_idx', 'o', configs.numStqEntries, configs.stpAddrW)

    ga_ls_order_o      = LogicVecArray('ga_ls_order', 'o', configs.numLdqEntries, configs.numStqEntries)

    # The number of empty load and store is calculated with cyclic subtraction.
    # If the empty signal is high, then set the number to max value.
    loads_sub    = LogicVec('loads_sub', 'w', configs.ldqAddrW)
    stores_sub   = LogicVec('stores_sub', 'w', configs.stqAddrW)
    empty_loads  = LogicVec('empty_loads', 'w', configs.emptyLdAddrW)
    empty_stores = LogicVec('empty_stores', 'w', configs.emptyStAddrW)

    arch += WrapSub(loads_sub, ldq_head_i, ldq_tail_i, configs.numLdqEntries)
    arch += WrapSub(stores_sub, stq_head_i, stq_tail_i, configs.numStqEntries)

    arch += Op(empty_loads, configs.numLdqEntries, 'when', ldq_empty_i, 'else', \
        '(', '\'0\'', '&', loads_sub, ')')
    arch += Op(empty_stores, configs.numStqEntries, 'when', stq_empty_i, 'else', \
        '(', '\'0\'', '&', stores_sub, ')')

    # Generate handshake signals
    group_init_ready = LogicArray('group_init_ready', 'w', configs.numGroups)
    group_init_hs    = LogicArray('group_init_hs', 'w', configs.numGroups)

    for i in range(0, configs.numGroups):
        arch += Op(group_init_ready[i], \
            '\'1\'', 'when',
            '(', empty_loads,  '>=', (configs.gaNumLoads[i], configs.emptyLdAddrW),  ')', 'and', \
            '(', empty_stores, '>=', (configs.gaNumStores[i], configs.emptyStAddrW), ')',
            'else', '\'0\'')

    if (configs.gaMulti):
        group_init_and = LogicArray('group_init_and', 'w', configs.numGroups)
        ga_rr_mask     = LogicVec('ga_rr_mask', 'r', configs.numGroups)
        ga_rr_mask.regInit()
        for i in range(0, configs.numGroups):
            arch += Op(group_init_and[i], group_init_ready[i], 'and', group_init_valid_i[i])
            arch += Op(group_init_ready_o[i], group_init_hs[i])
        arch += CyclicPriorityMasking(group_init_hs, group_init_and, ga_rr_mask)
        for i in range(0, configs.numGroups):
            arch += Op((ga_rr_mask, (i+1) % configs.numGroups), (group_init_hs, i))
    else:
        for i in range(0, configs.numGroups):
            arch += Op(group_init_ready_o[i], group_init_ready[i])
            arch += Op(group_init_hs[i], group_init_ready[i], 'and', group_init_valid_i[i])

    # ROM value
    if (configs.ldpAddrW > 0):
        ldq_port_idx_rom = LogicVecArray('ldq_port_idx_rom', 'w', configs.numLdqEntries, configs.ldpAddrW)
    if (configs.stpAddrW > 0):
        stq_port_idx_rom = LogicVecArray('stq_port_idx_rom', 'w', configs.numStqEntries, configs.stpAddrW)
    ga_ls_order_rom  = LogicVecArray('ga_ls_order_rom', 'w', configs.numLdqEntries, configs.numStqEntries)
    ga_ls_order_temp = LogicVecArray('ga_ls_order_temp', 'w', configs.numLdqEntries, configs.numStqEntries)
    if (configs.ldpAddrW > 0):
        arch += Mux1HROM(ldq_port_idx_rom, configs.gaLdPortIdx, group_init_hs)
    if (configs.stpAddrW > 0):
        arch += Mux1HROM(stq_port_idx_rom, configs.gaStPortIdx, group_init_hs)
    arch += Mux1HROM(ga_ls_order_rom, configs.gaLdOrder, group_init_hs, MaskLess)
    arch += Mux1HROM(num_loads, configs.gaNumLoads, group_init_hs)
    arch += Mux1HROM(num_stores, configs.gaNumStores, group_init_hs)
    arch += Op(num_loads_o, num_loads)
    arch += Op(num_stores_o, num_stores)

    ldq_wen_unshifted = LogicArray('ldq_wen_unshifted', 'w', configs.numLdqEntries)
    stq_wen_unshifted = LogicArray('stq_wen_unshifted', 'w', configs.numStqEntries)
    for i in range(0, configs.numLdqEntries):
        arch += Op(ldq_wen_unshifted[i],
            '\'1\'', 'when',
            num_loads, '>', (i, configs.ldqAddrW),
            'else', '\'0\''
        )
    for i in range(0, configs.numStqEntries):
        arch += Op(stq_wen_unshifted[i],
            '\'1\'', 'when',
            num_stores, '>', (i, configs.stqAddrW),
            'else', '\'0\''
        )
    
    # Shift the arrays
    if (configs.ldpAddrW > 0):
        arch += CyclicLeftShift(ldq_port_idx_o, ldq_port_idx_rom, ldq_tail_i)
    if (configs.stpAddrW > 0):
        arch += CyclicLeftShift(stq_port_idx_o, stq_port_idx_rom, stq_tail_i)
    arch += CyclicLeftShift(ldq_wen_o, ldq_wen_unshifted, ldq_tail_i)
    arch += CyclicLeftShift(stq_wen_o, stq_wen_unshifted, stq_tail_i)
    for i in range(0, configs.numLdqEntries):
        arch += CyclicLeftShift(ga_ls_order_temp[i], ga_ls_order_rom[i], stq_tail_i)
    arch += CyclicLeftShift(ga_ls_order_o, ga_ls_order_temp, ldq_tail_i)

    ######   Write To File  ######
    portInitString += '\n\t);'
    if (configs.gaMulti):
        regInitString += '\tend process;\n'
    else:
        regInitString = ''

    # Write to the file
    with open(f'{path_rtl}/{name}_core.vhd', 'a') as file:
        file.write('\n\n')
        file.write(library)
        file.write(f'entity {name + suffix} is\n')
        file.write(portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name + suffix} is\n')
        file.write(signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write(regInitString + 'end architecture;\n')
    return

def PortToQueueDispatcherInit(
    name:               str,
    numPorts:           int,
    numEntries:         int,
    port_bits_i:        LogicVecArray,
    port_valid_i:       LogicArray,
    port_ready_o:       LogicArray,
    entry_valid_i:      LogicArray,
    entry_bits_valid_i: LogicArray,
    entry_port_idx_i:   LogicVecArray,
    entry_bits_o:       LogicVecArray,
    entry_wen_o:        LogicArray,
    queue_head_oh_i:    LogicVec
) -> str:
    global tabLevel
    arch = '\t' * tabLevel + f'{name}_dispatcher : entity work.{name}\n'
    tabLevel += 1
    arch += '\t' * tabLevel + f'port map(\n'
    tabLevel += 1
    arch += '\t' * tabLevel + f'rst => rst,\n'
    arch += '\t' * tabLevel + f'clk => clk,\n'
    for i in range(0, numPorts):
        arch += '\t' * tabLevel + f'port_bits_{i}_i => {port_bits_i.getNameRead(i)},\n'
    for i in range(0, numPorts):
        arch += '\t' * tabLevel + f'port_ready_{i}_o => {port_ready_o.getNameWrite(i)},\n'
    for i in range(0, numPorts):
        arch += '\t' * tabLevel + f'port_valid_{i}_i => {port_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_valid_{i}_i => {entry_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_bits_valid_{i}_i => {entry_bits_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (numPorts != 1):
            arch += '\t' * tabLevel + f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_bits_{i}_o => {entry_bits_o.getNameWrite(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_wen_{i}_o => {entry_wen_o.getNameWrite(i)},\n'
    arch += '\t' * tabLevel + f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
    tabLevel -= 1
    arch += '\t' * tabLevel + f');\n'
    tabLevel -= 1
    return arch

def QueueToPortDispatcherInit(
    name:               str,
    numPorts:           int,
    numEntries:         int,
    port_bits_o:        LogicVecArray,
    port_valid_o:       LogicArray,
    port_ready_i:       LogicArray,
    entry_valid_i:      LogicArray,
    entry_bits_valid_i: LogicArray,
    entry_port_idx_i:   LogicVecArray,
    entry_bits_i:       LogicVecArray,
    entry_reset_o:      LogicArray,
    queue_head_oh_i:    LogicVec
) -> str:
    global tabLevel
    arch = '\t' * tabLevel + f'{name}_dispatcher : entity work.{name}\n'
    tabLevel += 1
    arch += '\t' * tabLevel + f'port map(\n'
    tabLevel += 1
    arch += '\t' * tabLevel + f'rst => rst,\n'
    arch += '\t' * tabLevel + f'clk => clk,\n'
    for i in range(0, numPorts):
        if (port_bits_o != None):
            arch += '\t' * tabLevel + f'port_bits_{i}_o => {port_bits_o.getNameWrite(i)},\n'
    for i in range(0, numPorts):
        arch += '\t' * tabLevel + f'port_ready_{i}_i => {port_ready_i.getNameRead(i)},\n'
    for i in range(0, numPorts):
        arch += '\t' * tabLevel + f'port_valid_{i}_o => {port_valid_o.getNameWrite(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_valid_{i}_i => {entry_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_bits_valid_{i}_i => {entry_bits_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (numPorts != 1):
            arch += '\t' * tabLevel + f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (entry_bits_i != None):
            arch += '\t' * tabLevel + f'entry_bits_{i}_i => {entry_bits_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += '\t' * tabLevel + f'entry_reset_{i}_o => {entry_reset_o.getNameWrite(i)},\n'
    arch += '\t' * tabLevel + f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
    tabLevel -= 1
    arch += '\t' * tabLevel + f');\n'
    tabLevel -= 1
    return arch

def GroupAllocatorInit(
    name:               str,
    configs:            Configs,
    group_init_valid_i: LogicArray,
    group_init_ready_o: LogicArray,
    ldq_tail_i:         LogicVec,
    ldq_head_i:         LogicVec,
    ldq_empty_i:        Logic,
    stq_tail_i:         LogicVec,
    stq_head_i:         LogicVec,
    stq_empty_i:        Logic,
    ldq_wen_o:          LogicArray,
    num_loads_o:        LogicVec,
    ldq_port_idx_o:     LogicVecArray,
    stq_wen_o:          LogicArray,
    num_stores_o:       LogicVec,
    stq_port_idx_o:     LogicVecArray,
    ga_ls_order_o:      LogicVecArray
) -> str:
    global tabLevel
    arch = '\t' * tabLevel + f'{name} : entity work.{name}\n'
    tabLevel += 1
    arch += '\t' * tabLevel + f'port map(\n'
    tabLevel += 1

    arch += '\t' * tabLevel + f'rst => rst,\n'
    arch += '\t' * tabLevel + f'clk => clk,\n'

    for i in range(0, configs.numGroups):
        arch += '\t' * tabLevel + f'group_init_valid_{i}_i => {group_init_valid_i.getNameRead(i)},\n'
    for i in range(0, configs.numGroups):
        arch += '\t' * tabLevel + f'group_init_ready_{i}_o => {group_init_ready_o.getNameWrite(i)},\n'

    arch += '\t' * tabLevel + f'ldq_tail_i => {ldq_tail_i.getNameRead()},\n'
    arch += '\t' * tabLevel + f'ldq_head_i => {ldq_head_i.getNameRead()},\n'
    arch += '\t' * tabLevel + f'ldq_empty_i => {ldq_empty_i.getNameRead()},\n'

    arch += '\t' * tabLevel + f'stq_tail_i => {stq_tail_i.getNameRead()},\n'
    arch += '\t' * tabLevel + f'stq_head_i => {stq_head_i.getNameRead()},\n'
    arch += '\t' * tabLevel + f'stq_empty_i => {stq_empty_i.getNameRead()},\n'

    for i in range(0, configs.numLdqEntries):
        arch += '\t' * tabLevel + f'ldq_wen_{i}_o => {ldq_wen_o.getNameWrite(i)},\n'
    arch += '\t' * tabLevel + f'num_loads_o => {num_loads_o.getNameWrite()},\n'
    if (configs.ldpAddrW > 0):
        for i in range(0, configs.numLdqEntries):
            arch += '\t' * tabLevel + f'ldq_port_idx_{i}_o => {ldq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, configs.numStqEntries):
        arch += '\t' * tabLevel + f'stq_wen_{i}_o => {stq_wen_o.getNameWrite(i)},\n'
    if (configs.stpAddrW > 0):
        for i in range(0, configs.numStqEntries):
            arch += '\t' * tabLevel + f'stq_port_idx_{i}_o => {stq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, configs.numLdqEntries):
        arch += '\t' * tabLevel + f'ga_ls_order_{i}_o => {ga_ls_order_o.getNameWrite(i)},\n'

    arch += '\t' * tabLevel + f'num_stores_o => {num_stores_o.getNameWrite()}\n'

    tabLevel -= 1
    arch += '\t' * tabLevel + f');\n'
    tabLevel -= 1
    return arch

def LSQ(path_rtl: str, name: str, configs: Configs):

    # Initialize the global parameters
    global tabLevel
    global tempCount
    global signalInitString
    global portInitString
    global regInitString
    global library
    tabLevel = 1
    tempCount = 0
    signalInitString = ''
    portInitString   = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    regInitString    = '\tprocess (clk, rst) is\n' + '\tbegin\n'
    arch             = ''

    ###### LSQ Architecture ######
    ######        IOs       ######

    # group initialzation signals
    group_init_valid_i = LogicArray('group_init_valid', 'i', configs.numGroups)
    group_init_ready_o = LogicArray('group_init_ready', 'o', configs.numGroups)

    # Memory access ports, i.e., the connection "kernel -> LSQ"
    # Load address channel (addr, valid, ready) from kernel, contains signals:
    ldp_addr_i         = LogicVecArray('ldp_addr', 'i', configs.numLdPorts, configs.addrW)
    ldp_addr_valid_i   = LogicArray('ldp_addr_valid', 'i', configs.numLdPorts)
    ldp_addr_ready_o   = LogicArray('ldp_addr_ready', 'o', configs.numLdPorts)

    # Load data channel (data, valid, ready) to kernel
    ldp_data_o         = LogicVecArray('ldp_data', 'o', configs.numLdPorts, configs.dataW)
    ldp_data_valid_o   = LogicArray('ldp_data_valid', 'o', configs.numLdPorts)
    ldp_data_ready_i   = LogicArray('ldp_data_ready', 'i', configs.numLdPorts)

    # Store address channel (addr, valid, ready) from kernel
    stp_addr_i         = LogicVecArray('stp_addr', 'i', configs.numStPorts, configs.addrW)
    stp_addr_valid_i   = LogicArray('stp_addr_valid', 'i', configs.numStPorts)
    stp_addr_ready_o   = LogicArray('stp_addr_ready', 'o', configs.numStPorts)

    # Store data channel (data, valid, ready) from kernel
    stp_data_i         = LogicVecArray('stp_data', 'i', configs.numStPorts, configs.dataW)
    stp_data_valid_i   = LogicArray('stp_data_valid', 'i', configs.numStPorts)
    stp_data_ready_o   = LogicArray('stp_data_ready', 'o', configs.numStPorts)

    if configs.stResp:
        stp_exec_valid_o   = LogicArray('stp_exec_valid', 'o', configs.numStPorts)
        stp_exec_ready_i   = LogicArray('stp_exec_ready', 'i', configs.numStPorts)

    # queue empty signal
    empty_o            = Logic('empty', 'o')

    # Memory interface: i.e., the connection LSQ -> AXI
    # We assume that the memory interface has
    # 1. A read request channel (rreq) and a read response channel (rresp).
    # 2. A write request channel (wreq) and a write response channel (wresp).
    rreq_valid_o       = LogicArray('rreq_valid', 'o', configs.numLdMem)
    rreq_ready_i       = LogicArray('rreq_ready', 'i', configs.numLdMem)
    rreq_id_o          = LogicVecArray('rreq_id', 'o', configs.numLdMem, configs.idW)
    rreq_addr_o        = LogicVecArray('rreq_addr', 'o', configs.numLdMem, configs.addrW)

    rresp_valid_i      = LogicArray('rresp_valid', 'i', configs.numLdMem)
    rresp_ready_o      = LogicArray('rresp_ready', 'o', configs.numLdMem)
    rresp_id_i         = LogicVecArray('rresp_id', 'i', configs.numLdMem, configs.idW)
    rresp_data_i       = LogicVecArray('rresp_data', 'i', configs.numLdMem, configs.dataW)

    wreq_valid_o       = LogicArray('wreq_valid', 'o', configs.numStMem)
    wreq_ready_i       = LogicArray('wreq_ready', 'i', configs.numStMem)
    wreq_id_o          = LogicVecArray('wreq_id', 'o', configs.numStMem, configs.idW)
    wreq_addr_o        = LogicVecArray('wreq_addr', 'o', configs.numStMem, configs.addrW)
    wreq_data_o        = LogicVecArray('wreq_data', 'o', configs.numStMem, configs.dataW)

    wresp_valid_i      = LogicArray('wresp_valid', 'i', configs.numStMem)
    wresp_ready_o      = LogicArray('wresp_ready', 'o', configs.numStMem)
    wresp_id_i         = LogicVecArray('wresp_id', 'i', configs.numStMem, configs.idW)
    
    #! If this is the lsq master, then we need the following logic
    #! Define new interfaces needed by dynamatic
    if (configs.master):
      memStart_ready = Logic('memStart_ready', 'o')
      memStart_valid = Logic('memStart_valid', 'i')
      ctrlEnd_ready = Logic('ctrlEnd_ready', 'o')
      ctrlEnd_valid = Logic('ctrlEnd_valid', 'i')
      memEnd_ready = Logic('memEnd_ready', 'i')
      memEnd_valid = Logic('memEnd_valid', 'o')
      
      #! Add extra signals required
      memStartReady = Logic('memStartReady', 'w')
      memEndValid = Logic('memEndValid', 'w')
      ctrlEndReady = Logic('ctrlEndReady', 'w')
      temp_gen_mem = Logic('TEMP_GEN_MEM', 'w') 
      
      #! Define the needed logic
      arch += "\t-- Define the intermediate logic\n"
      arch += f"\tTEMP_GEN_MEM <= {ctrlEnd_valid.getNameRead()} and stq_empty and ldq_empty;\n"
      
      arch += "\t-- Define logic for the new interfaces needed by dynamatic\n"
      arch += "\tprocess (clk) is\n\tbegin\n"
      arch += '\t' * 2 + "if rising_edge(clk) then\n"
      arch += '\t' * 3 + "if rst = '1' then\n"
      arch += '\t' * 4 + "memStartReady <= '1';\n"
      arch += '\t' * 4 + "memEndValid <= '0';\n"
      arch += '\t' * 4 + "ctrlEndReady <= '0';\n"
      arch += '\t' * 3 + "else\n"
      arch += '\t' * 4 + "memStartReady <= (memEndValid and memEnd_ready_i) or ((not (memStart_valid_i and memStartReady)) and memStartReady);\n"
      arch += '\t' * 4 + "memEndValid <= TEMP_GEN_MEM or memEndValid;\n"
      arch += '\t' * 4 + "ctrlEndReady <= (not (ctrlEnd_valid_i and ctrlEndReady)) and (TEMP_GEN_MEM or ctrlEndReady);\n"
      arch += '\t' * 3 + "end if;\n"
      arch += '\t' * 2 + "end if;\n"
      arch += "\tend process;\n\n"
      
      #! Assign signals for the newly added ports
      arch += "\t-- Update new memory interfaces\n"
      arch += Op(memStart_ready, memStartReady)
      arch += Op(ctrlEnd_ready, ctrlEndReady)
      arch += Op(memEnd_valid, memEndValid)

    ######  Queue Registers ######
    # Load Queue Entries
    ldq_valid      = LogicArray('ldq_valid', 'r', configs.numLdqEntries)
    ldq_issue      = LogicArray('ldq_issue', 'r', configs.numLdqEntries)
    if (configs.ldpAddrW > 0):
        ldq_port_idx = LogicVecArray('ldq_port_idx', 'r', configs.numLdqEntries, configs.ldpAddrW)
    else:
        ldq_port_idx = None
    ldq_addr_valid = LogicArray('ldq_addr_valid', 'r', configs.numLdqEntries)
    ldq_addr       = LogicVecArray('ldq_addr', 'r', configs.numLdqEntries, configs.addrW)
    ldq_data_valid = LogicArray('ldq_data_valid', 'r', configs.numLdqEntries)
    ldq_data       = LogicVecArray('ldq_data', 'r', configs.numLdqEntries, configs.dataW)

    # Store Queue Entries
    stq_valid      = LogicArray('stq_valid', 'r', configs.numStqEntries)
    if configs.stResp:
        stq_exec   = LogicArray('stq_exec', 'r', configs.numStqEntries)
    if (configs.stpAddrW > 0):
        stq_port_idx = LogicVecArray('stq_port_idx', 'r', configs.numStqEntries, configs.stpAddrW)
    else:
        stq_port_idx = None
    stq_addr_valid = LogicArray('stq_addr_valid', 'r', configs.numStqEntries)
    stq_addr       = LogicVecArray('stq_addr', 'r', configs.numStqEntries, configs.addrW)
    stq_data_valid = LogicArray('stq_data_valid', 'r', configs.numStqEntries)
    stq_data       = LogicVecArray('stq_data', 'r', configs.numStqEntries, configs.dataW)

    # Order for load-store
    store_is_older = LogicVecArray('store_is_older', 'r', configs.numLdqEntries, configs.numStqEntries)

    # Pointers
    ldq_tail       = LogicVec('ldq_tail', 'r', configs.ldqAddrW)
    ldq_head       = LogicVec('ldq_head', 'r', configs.ldqAddrW)

    stq_tail       = LogicVec('stq_tail', 'r', configs.stqAddrW)
    stq_head       = LogicVec('stq_head', 'r', configs.stqAddrW)
    stq_issue      = LogicVec('stq_issue', 'r', configs.stqAddrW)
    stq_resp       = LogicVec('stq_resp', 'r', configs.stqAddrW)

    # Entry related signals
    # From port dispatchers
    ldq_wen        = LogicArray('ldq_wen', 'w', configs.numLdqEntries)
    ldq_addr_wen   = LogicArray('ldq_addr_wen', 'w', configs.numLdqEntries)
    ldq_reset      = LogicArray('ldq_reset', 'w', configs.numLdqEntries)
    stq_wen        = LogicArray('stq_wen', 'w', configs.numStqEntries)
    stq_addr_wen   = LogicArray('stq_addr_wen', 'w', configs.numStqEntries)
    stq_data_wen   = LogicArray('stq_data_wen', 'w', configs.numStqEntries)
    stq_reset  = LogicArray('stq_reset', 'w', configs.numStqEntries)
    # From Read/Write Block
    ldq_data_wen   = LogicArray('ldq_data_wen', 'w', configs.numLdqEntries)
    ldq_issue_set  = LogicArray('ldq_issue_set', 'w', configs.numLdqEntries)
    if configs.stResp:
        stq_exec_set   = LogicArray('stq_exec_set', 'w', configs.numStqEntries)
    # Form Group Allocator
    ga_ls_order    = LogicVecArray('ga_ls_order', 'w', configs.numLdqEntries, configs.numStqEntries)

    # Pointer related signals
    # For updating pointers
    num_loads      = LogicVec('num_loads', 'w', configs.ldqAddrW)
    num_stores     = LogicVec('num_stores', 'w', configs.stqAddrW)
    stq_issue_en   = Logic('stq_issue_en', 'w')
    stq_resp_en    = Logic('stq_resp_en', 'w')
    # Generated by pointers
    ldq_empty      = Logic('ldq_empty', 'w')
    stq_empty      = Logic('stq_empty', 'w')
    ldq_head_oh    = LogicVec('ldq_head_oh', 'w', configs.numLdqEntries)
    stq_head_oh    = LogicVec('stq_head_oh', 'w', configs.numStqEntries)

    arch += BitsToOH(ldq_head_oh, ldq_head)
    arch += BitsToOH(stq_head_oh, stq_head)

    # update queue entries
    # load queue
    if configs.pipe0 or configs.pipeComp:
        ldq_wen_p0 = LogicArray('ldq_wen_p0', 'r', configs.numLdqEntries)
        ldq_wen_p0.regInit()
        if configs.pipe0 and configs.pipeComp:
            ldq_wen_p1 = LogicArray('ldq_wen_p1', 'r', configs.numLdqEntries)
            ldq_wen_p1.regInit()
    ldq_valid_next = LogicArray('ldq_valid_next', 'w', configs.numLdqEntries)
    for i in range(0, configs.numLdqEntries):
        arch += Op(ldq_valid_next[i],
            'not', ldq_reset[i], 'and', ldq_valid[i]
        )
        arch += Op(ldq_valid[i],
            ldq_wen[i], 'or', ldq_valid_next[i]
        )
        if configs.pipe0 or configs.pipeComp:
            arch += Op(ldq_wen_p0[i], ldq_wen[i])
            if configs.pipe0 and configs.pipeComp:
                arch += Op(ldq_wen_p1[i], ldq_wen[i])
                arch += Op(ldq_issue[i],
                    'not', ldq_wen_p1[i], 'and',
                    '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
                )
            else:
                arch += Op(ldq_issue[i],
                    'not', ldq_wen_p0[i], 'and',
                    '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
                )
        else:
            arch += Op(ldq_issue[i],
                'not', ldq_wen[i], 'and',
                '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
            )
        arch += Op(ldq_addr_valid[i],
            'not', ldq_wen[i], 'and',
            '(', ldq_addr_wen[i], 'or', ldq_addr_valid[i], ')'
        )
        arch += Op(ldq_data_valid[i],
            'not', ldq_wen[i], 'and',
            '(', ldq_data_wen[i], 'or', ldq_data_valid[i], ')'
        )
    # store queue
    stq_valid_next = LogicArray('stq_valid_next', 'w', configs.numStqEntries)
    for i in range(0, configs.numStqEntries):
        arch += Op(stq_valid_next[i],
            'not', stq_reset[i], 'and', stq_valid[i]
        )
        arch += Op(stq_valid[i],
            stq_wen[i], 'or', stq_valid_next[i]
        )
        if configs.stResp:
            arch += Op(stq_exec[i],
                'not', stq_wen[i], 'and',
                '(', stq_exec_set[i], 'or', stq_exec[i], ')'
            )
        arch += Op(stq_addr_valid[i],
            'not', stq_wen[i], 'and',
            '(', stq_addr_wen[i], 'or', stq_addr_valid[i], ')'
        )
        arch += Op(stq_data_valid[i],
            'not', stq_wen[i], 'and',
            '(', stq_data_wen[i], 'or', stq_data_valid[i], ')'
        )


    # order matrix
    # store_is_older(i,j) = (not stq_reset(j) and (stq_valid(j) or ga_ls_order(i, j))) 
    #                  when ldq_wen(i)
    #                  else not stq_reset(j) and store_is_older(i, j)
    for i in range(0, configs.numLdqEntries):
        for j in range(0, configs.numStqEntries):
            arch += Op((store_is_older, i, j),
                '(', 'not', (stq_reset, j), 'and', '(', (stq_valid, j), 'or', (ga_ls_order, i, j), ')', ')',
                'when', (ldq_wen, i), 'else',
                'not', (stq_reset, j), 'and', (store_is_older, i, j)
            )

    # pointers update
    ldq_not_empty = Logic('ldq_not_empty', 'w')
    stq_not_empty = Logic('stq_not_empty', 'w')
    arch += Reduce(ldq_not_empty, ldq_valid, 'or')
    arch += Op(ldq_empty, 'not', ldq_not_empty)
    arch += MuxLookUp(stq_not_empty, stq_valid, stq_head)
    arch += Op(stq_empty, 'not', stq_not_empty)
    arch += Op(empty_o, ldq_empty, 'and', stq_empty)

    arch += WrapAdd(ldq_tail, ldq_tail, num_loads, configs.numLdqEntries)
    arch += WrapAdd(stq_tail, stq_tail, num_stores, configs.numStqEntries)
    arch += WrapAddConst(stq_issue, stq_issue, 1, configs.numStqEntries)
    arch += WrapAddConst(stq_resp, stq_resp, 1, configs.numStqEntries)

    ldq_tail_oh      = LogicVec('ldq_tail_oh', 'w', configs.numLdqEntries)
    arch += BitsToOH(ldq_tail_oh, ldq_tail)
    ldq_head_next_oh = LogicVec('ldq_head_next_oh', 'w', configs.numLdqEntries)
    ldq_head_next    = LogicVec('ldq_head_next', 'w', configs.ldqAddrW)
    ldq_head_sel     = Logic('ldq_head_sel', 'w')
    if configs.headLag:
        # Update the head pointer according to the valid signal of last cycle
        arch += CyclicPriorityMasking(ldq_head_next_oh, ldq_valid, ldq_tail_oh)
        arch += Reduce(ldq_head_sel, ldq_valid, 'or')
    else:
        arch += CyclicPriorityMasking(ldq_head_next_oh, ldq_valid_next, ldq_tail_oh)
        arch += Reduce(ldq_head_sel, ldq_valid_next, 'or')
    arch += OHToBits(ldq_head_next, ldq_head_next_oh)
    arch += Op(ldq_head, ldq_head_next, 'when', ldq_head_sel, 'else', ldq_tail)
    
    stq_tail_oh      = LogicVec('stq_tail_oh', 'w', configs.numStqEntries)
    arch += BitsToOH(stq_tail_oh, stq_tail)
    stq_head_next_oh = LogicVec('stq_head_next_oh', 'w', configs.numStqEntries)
    stq_head_next    = LogicVec('stq_head_next', 'w', configs.stqAddrW)
    stq_head_sel     = Logic('stq_head_sel', 'w')
    if configs.stResp:
        if configs.headLag:
            # Update the head pointer according to the valid signal of last cycle
            arch += CyclicPriorityMasking(stq_head_next_oh, stq_valid, stq_tail_oh)
            arch += Reduce(stq_head_sel, stq_valid, 'or')
        else:
            arch += CyclicPriorityMasking(stq_head_next_oh, stq_valid_next, stq_tail_oh)
            arch += Reduce(stq_head_sel, stq_valid_next, 'or')
        arch += OHToBits(stq_head_next, stq_head_next_oh)
        arch += Op(stq_head, stq_head_next, 'when', stq_head_sel, 'else', stq_tail)
    else:
        arch += WrapAddConst(stq_head_next, stq_head, 1, configs.numStqEntries)
        arch += Op(stq_head_sel, wresp_valid_i[0])
        arch += Op(stq_head, stq_head_next, 'when', stq_head_sel, 'else', stq_head)
        

    # Load Queue Entries
    ldq_valid.regInit(init=[0]*configs.numLdqEntries)
    ldq_issue.regInit()
    if (configs.ldpAddrW > 0):
        ldq_port_idx.regInit(ldq_wen)
    ldq_addr_valid.regInit()
    ldq_addr.regInit(ldq_addr_wen)
    ldq_data_valid.regInit()
    ldq_data.regInit(ldq_data_wen)

    # Store Queue Entries
    stq_valid.regInit(init=[0]*configs.numStqEntries)
    if configs.stResp:
        stq_exec.regInit()
    if (configs.stpAddrW > 0):
        stq_port_idx.regInit(stq_wen)
    stq_addr_valid.regInit()
    stq_addr.regInit(stq_addr_wen)
    stq_data_valid.regInit()
    stq_data.regInit(stq_data_wen)

    # Order for load-store
    store_is_older.regInit()

    # Pointers
    ldq_tail.regInit(init=0)
    ldq_head.regInit(init=0)

    stq_tail.regInit(init=0)
    stq_head.regInit(init=0)
    stq_issue.regInit(enable=stq_issue_en, init=0)
    stq_resp.regInit(enable=stq_resp_en, init=0)

    ######   Entity Init    ######

    # Group Allocator
    arch += GroupAllocatorInit(name + '_ga', configs,
        group_init_valid_i, group_init_ready_o,
        ldq_tail, ldq_head, ldq_empty,
        stq_tail, stq_head, stq_empty,
        ldq_wen, num_loads, ldq_port_idx,
        stq_wen, num_stores, stq_port_idx,
        ga_ls_order
    )

    # Load Address Port Dispatcher
    arch += PortToQueueDispatcherInit(name + '_lda',
        configs.numLdPorts, configs.numLdqEntries,
        ldp_addr_i, ldp_addr_valid_i, ldp_addr_ready_o,
        ldq_valid, ldq_addr_valid, ldq_port_idx, ldq_addr, ldq_addr_wen, ldq_head_oh
    )
    # Load Data Port Dispatcher
    arch += QueueToPortDispatcherInit(name + '_ldd',
        configs.numLdPorts, configs.numLdqEntries,
        ldp_data_o, ldp_data_valid_o, ldp_data_ready_i,
        ldq_valid, ldq_data_valid, ldq_port_idx, ldq_data, ldq_reset, ldq_head_oh
    )    
    # Store Address Port Dispatcher
    arch += PortToQueueDispatcherInit(name + '_sta',
        configs.numStPorts, configs.numStqEntries,
        stp_addr_i, stp_addr_valid_i, stp_addr_ready_o,
        stq_valid, stq_addr_valid, stq_port_idx, stq_addr, stq_addr_wen, stq_head_oh
    )
    # Store Data Port Dispatcher
    arch += PortToQueueDispatcherInit(name + '_std',
        configs.numStPorts, configs.numStqEntries,
        stp_data_i, stp_data_valid_i, stp_data_ready_o,
        stq_valid, stq_data_valid, stq_port_idx, stq_data, stq_data_wen, stq_head_oh
    )
    # Store Backward Port Dispatcher
    if configs.stResp:
        arch += QueueToPortDispatcherInit(name + '_stb',
            configs.numStPorts, configs.numStqEntries,
            None, stp_exec_valid_o, stp_exec_ready_i,
            stq_valid, stq_exec, stq_port_idx, None, stq_reset, stq_head_oh
        )

    if configs.pipe0:
        ###### Dependency Check ######
        load_idx_oh    = LogicVecArray('load_idx_oh', 'w', configs.numLdMem, configs.numLdqEntries)
        load_en        = LogicArray('load_en', 'w', configs.numLdMem)

        assert(configs.numStMem == 1) # Multiple store channels not yet implemented
        store_idx      = LogicVec('store_idx', 'w', configs.stqAddrW)
        store_en       = Logic('store_en', 'w')
        
        bypass_idx_oh_p0 = LogicVecArray('bypass_idx_oh_p0', 'r', configs.numLdqEntries, configs.numStqEntries)
        bypass_idx_oh_p0.regInit()
        bypass_en      = LogicArray('bypass_en', 'w', configs.numLdqEntries)

        # Matrix Generation
        ld_st_conflict = LogicVecArray('ld_st_conflict', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass     = LogicVecArray('can_bypass', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass_p0  = LogicVecArray('can_bypass_p0', 'r', configs.numLdqEntries, configs.numStqEntries)
        can_bypass_p0.regInit(init=[0]*configs.numLdqEntries)

        if configs.pipeComp:
            ldq_valid_pcomp      = LogicArray('ldq_valid_pcomp', 'r', configs.numLdqEntries)
            ldq_addr_valid_pcomp = LogicArray('ldq_addr_valid_pcomp', 'r', configs.numLdqEntries)
            stq_valid_pcomp      = LogicArray('stq_valid_pcomp', 'r', configs.numStqEntries)
            stq_addr_valid_pcomp = LogicArray('stq_addr_valid_pcomp', 'r', configs.numStqEntries)
            stq_data_valid_pcomp = LogicArray('stq_data_valid_pcomp', 'r', configs.numStqEntries)
            addr_valid_pcomp     = LogicVecArray('addr_valid_pcomp', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same_pcomp      = LogicVecArray('addr_same_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            store_is_older_pcomp = LogicVecArray('store_is_older_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            
            ldq_valid_pcomp.regInit(init=[0]*configs.numLdqEntries)
            ldq_addr_valid_pcomp.regInit()
            stq_valid_pcomp.regInit(init=[0]*configs.numStqEntries)
            stq_addr_valid_pcomp.regInit()
            stq_data_valid_pcomp.regInit()
            addr_same_pcomp.regInit()
            store_is_older_pcomp.regInit()

            for i in range(0, configs.numLdqEntries):
                arch += Op((ldq_valid_pcomp, i), (ldq_valid, i))
                arch += Op((ldq_addr_valid_pcomp, i), (ldq_addr_valid, i))
            for j in range(0, configs.numStqEntries):
                arch += Op((stq_valid_pcomp, j), (stq_valid, j))
                arch += Op((stq_addr_valid_pcomp, j), (stq_addr_valid, j))
                arch += Op((stq_data_valid_pcomp, j), (stq_data_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((store_is_older_pcomp, i, j), (store_is_older, i, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_valid_pcomp, i, j), (ldq_addr_valid_pcomp, i), 'and', (stq_addr_valid_pcomp, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_same_pcomp, i, j), '\'1\'', 'when', (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')
                
            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (ld_st_conflict, i, j),
                        (stq_valid_pcomp, j),   'and',
                        (store_is_older_pcomp, i, j), 'and',
                        '(', (addr_same_pcomp, i, j), 'or', 'not', (stq_addr_valid_pcomp, j), ')'
                    )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass_p0, i, j),
                        (ldq_valid_pcomp, i),        'and',
                        (stq_data_valid_pcomp, j),   'and',
                        (addr_same_pcomp, i, j),     'and',
                        (addr_valid_pcomp, i, j)
                    )
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass, i, j),
                        'not', (ldq_issue, i), 'and',
                        (can_bypass_p0, i, j)
                    )

            # Load

            load_conflict    = LogicArray('load_conflict', 'w', configs.numLdqEntries)
            load_req_valid   = LogicArray('load_req_valid', 'w', configs.numLdqEntries)
            can_load         = LogicArray('can_load', 'w', configs.numLdqEntries)
            can_load_p0      = LogicArray('can_load_p0', 'r', configs.numLdqEntries)
            can_load_p0.regInit(init=[0]*configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(load_req_valid[i], ldq_valid_pcomp[i], 'and', ldq_addr_valid_pcomp[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load_p0[i], 'not', load_conflict[i], 'and', load_req_valid[i])
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load[i], 'not', ldq_issue[i], 'and', can_load_p0[i])

            ldq_head_oh_p0 = LogicVec('ldq_head_oh_p0', 'r', configs.numLdqEntries)
            ldq_head_oh_p0.regInit()
            arch += Op(ldq_head_oh_p0, ldq_head_oh) 

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(load_idx_oh[w], can_load_list[w], ldq_head_oh_p0)
                arch += Reduce(load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(can_load_list[w+1][i], 'not', load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store
            stq_issue_en_p0        = Logic('stq_issue_en_p0', 'r')
            stq_issue_next         = LogicVec('stq_issue_next', 'w', configs.stqAddrW)

            store_conflict         = Logic('store_conflict', 'w')

            can_store_curr         = Logic('can_store_curr', 'w')
            st_ld_conflict_curr    = LogicVec('st_ld_conflict_curr', 'w', configs.numLdqEntries)
            store_valid_curr       = Logic('store_valid_curr', 'w')
            store_data_valid_curr  = Logic('store_data_valid_curr', 'w')
            store_addr_valid_curr  = Logic('store_addr_valid_curr', 'w')

            can_store_next         = Logic('can_store_next', 'w')
            st_ld_conflict_next    = LogicVec('st_ld_conflict_next', 'w', configs.numLdqEntries)
            store_valid_next       = Logic('store_valid_next', 'w')
            store_data_valid_next  = Logic('store_data_valid_next', 'w')
            store_addr_valid_next  = Logic('store_addr_valid_next', 'w')

            can_store_p0           = Logic('can_store_p0', 'r')
            st_ld_conflict_p0      = LogicVec('st_ld_conflict_p0', 'r', configs.numLdqEntries)

            stq_issue_en_p0.regInit(init=0)
            can_store_p0.regInit(init=0)
            st_ld_conflict_p0.regInit()

            arch += Op(stq_issue_en_p0, stq_issue_en)
            arch += WrapAddConst(stq_issue_next, stq_issue, 1, configs.numStqEntries)

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict_curr, i),
                    (ldq_valid_pcomp, i), 'and',
                    'not', MuxIndex(store_is_older_pcomp[i], stq_issue), 'and',
                    '(', MuxIndex(addr_same_pcomp[i], stq_issue), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                )
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict_next, i),
                    (ldq_valid_pcomp, i), 'and',
                    'not', MuxIndex(store_is_older_pcomp[i], stq_issue_next), 'and',
                    '(', MuxIndex(addr_same_pcomp[i], stq_issue_next), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                )
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(store_valid_curr, stq_valid_pcomp, stq_issue)
            arch += MuxLookUp(store_data_valid_curr, stq_data_valid_pcomp, stq_issue)
            arch += MuxLookUp(store_addr_valid_curr, stq_addr_valid_pcomp, stq_issue)
            arch += Op(can_store_curr,
                store_valid_curr, 'and',
                store_data_valid_curr, 'and',
                store_addr_valid_curr
            )
            arch += MuxLookUp(store_valid_next, stq_valid_pcomp, stq_issue_next)
            arch += MuxLookUp(store_data_valid_next, stq_data_valid_pcomp, stq_issue_next)
            arch += MuxLookUp(store_addr_valid_next, stq_addr_valid_pcomp, stq_issue_next)
            arch += Op(can_store_next,
                store_valid_next, 'and',
                store_data_valid_next, 'and',
                store_addr_valid_next
            )
            # Multiplex from current and next
            arch += Op(st_ld_conflict_p0, st_ld_conflict_next, 'when', stq_issue_en, 'else', st_ld_conflict_curr)
            arch += Op(can_store_p0, can_store_next, 'when', stq_issue_en, 'else', can_store_curr)
            # The store conflicts with any load
            arch += Reduce(store_conflict, st_ld_conflict_p0, 'or')
            arch += Op(store_en, 'not', store_conflict, 'and', can_store_p0)

            arch += Op(store_idx, stq_issue)

            # Bypass
            stq_last_oh = LogicVec('stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(bypass_en_vec, bypass_idx_oh_p0[i], 'and', can_bypass[i])
                arch += Reduce(bypass_en[i], bypass_en_vec, 'or')
        else:
            addr_valid     = LogicVecArray('addr_valid', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same      = LogicVecArray('addr_same', 'w', configs.numLdqEntries, configs.numStqEntries)

            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_valid, i, j), (ldq_addr_valid, i), 'and', (stq_addr_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_same, i, j), '\'1\'', 'when', (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (ld_st_conflict, i, j),
                        (stq_valid, j),         'and',
                        (store_is_older, i, j), 'and',
                        '(', (addr_same, i, j), 'or', 'not', (stq_addr_valid, j), ')'
                    )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass_p0, i, j),
                        (ldq_valid, i),        'and',
                        (stq_data_valid, j),   'and',
                        (addr_same, i, j),     'and',
                        (addr_valid, i, j)
                    )
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass, i, j),
                        'not', (ldq_issue, i), 'and',
                        (can_bypass_p0, i, j)
                    )

            # Load

            load_conflict    = LogicArray('load_conflict', 'w', configs.numLdqEntries)
            load_req_valid   = LogicArray('load_req_valid', 'w', configs.numLdqEntries)
            can_load         = LogicArray('can_load', 'w', configs.numLdqEntries)
            can_load_p0      = LogicArray('can_load_p0', 'r', configs.numLdqEntries)
            can_load_p0.regInit(init=[0]*configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(load_req_valid[i], ldq_valid[i], 'and', ldq_addr_valid[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load_p0[i], 'not', load_conflict[i], 'and', load_req_valid[i])
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load[i], 'not', ldq_issue[i], 'and', can_load_p0[i])

            ldq_head_oh_p0 = LogicVec('ldq_head_oh_p0', 'r', configs.numLdqEntries)
            ldq_head_oh_p0.regInit()
            arch += Op(ldq_head_oh_p0, ldq_head_oh) 

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(load_idx_oh[w], can_load_list[w], ldq_head_oh_p0)
                arch += Reduce(load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(can_load_list[w+1][i], 'not', load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store
            stq_issue_en_p0        = Logic('stq_issue_en_p0', 'r')
            stq_issue_next         = LogicVec('stq_issue_next', 'w', configs.stqAddrW)

            store_conflict         = Logic('store_conflict', 'w')

            can_store_curr         = Logic('can_store_curr', 'w')
            st_ld_conflict_curr    = LogicVec('st_ld_conflict_curr', 'w', configs.numLdqEntries)
            store_valid_curr       = Logic('store_valid_curr', 'w')
            store_data_valid_curr  = Logic('store_data_valid_curr', 'w')
            store_addr_valid_curr  = Logic('store_addr_valid_curr', 'w')

            can_store_next         = Logic('can_store_next', 'w')
            st_ld_conflict_next    = LogicVec('st_ld_conflict_next', 'w', configs.numLdqEntries)
            store_valid_next       = Logic('store_valid_next', 'w')
            store_data_valid_next  = Logic('store_data_valid_next', 'w')
            store_addr_valid_next  = Logic('store_addr_valid_next', 'w')

            can_store_p0           = Logic('can_store_p0', 'r')
            st_ld_conflict_p0      = LogicVec('st_ld_conflict_p0', 'r', configs.numLdqEntries)

            stq_issue_en_p0.regInit(init=0)
            can_store_p0.regInit(init=0)
            st_ld_conflict_p0.regInit()

            arch += Op(stq_issue_en_p0, stq_issue_en)
            arch += WrapAddConst(stq_issue_next, stq_issue, 1, configs.numStqEntries)

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict_curr, i),
                    (ldq_valid, i), 'and',
                    'not', MuxIndex(store_is_older[i], stq_issue), 'and',
                    '(', MuxIndex(addr_same[i], stq_issue), 'or', 'not', (ldq_addr_valid, i), ')'
                )
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict_next, i),
                    (ldq_valid, i), 'and',
                    'not', MuxIndex(store_is_older[i], stq_issue_next), 'and',
                    '(', MuxIndex(addr_same[i], stq_issue_next), 'or', 'not', (ldq_addr_valid, i), ')'
                )
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(store_valid_curr, stq_valid, stq_issue)
            arch += MuxLookUp(store_data_valid_curr, stq_data_valid, stq_issue)
            arch += MuxLookUp(store_addr_valid_curr, stq_addr_valid, stq_issue)
            arch += Op(can_store_curr,
                store_valid_curr, 'and',
                store_data_valid_curr, 'and',
                store_addr_valid_curr
            )
            arch += MuxLookUp(store_valid_next, stq_valid, stq_issue_next)
            arch += MuxLookUp(store_data_valid_next, stq_data_valid, stq_issue_next)
            arch += MuxLookUp(store_addr_valid_next, stq_addr_valid, stq_issue_next)
            arch += Op(can_store_next,
                store_valid_next, 'and',
                store_data_valid_next, 'and',
                store_addr_valid_next
            )
            # Multiplex from current and next
            arch += Op(st_ld_conflict_p0, st_ld_conflict_next, 'when', stq_issue_en, 'else', st_ld_conflict_curr)
            arch += Op(can_store_p0, can_store_next, 'when', stq_issue_en, 'else', can_store_curr)
            # The store conflicts with any load
            arch += Reduce(store_conflict, st_ld_conflict_p0, 'or')
            arch += Op(store_en, 'not', store_conflict, 'and', can_store_p0)

            arch += Op(store_idx, stq_issue)

            # Bypass
            stq_last_oh = LogicVec('stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(bypass_en_vec, bypass_idx_oh_p0[i], 'and', can_bypass[i])
                arch += Reduce(bypass_en[i], bypass_en_vec, 'or')
    else:
        ###### Dependency Check ######

        load_idx_oh    = LogicVecArray('load_idx_oh', 'w', configs.numLdMem, configs.numLdqEntries)
        load_en        = LogicArray('load_en', 'w', configs.numLdMem)

        assert(configs.numStMem == 1) # Multiple store channels not yet implemented
        store_idx      = LogicVec('store_idx', 'w', configs.stqAddrW)
        store_en       = Logic('store_en', 'w')
        
        bypass_idx_oh  = LogicVecArray('bypass_idx_oh', 'w', configs.numLdqEntries, configs.numStqEntries)
        bypass_en      = LogicArray('bypass_en', 'w', configs.numLdqEntries)

        # Matrix Generation
        ld_st_conflict = LogicVecArray('ld_st_conflict', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass     = LogicVecArray('can_bypass', 'w', configs.numLdqEntries, configs.numStqEntries)

        if configs.pipeComp:
            ldq_valid_pcomp      = LogicArray('ldq_valid_pcomp', 'r', configs.numLdqEntries)
            ldq_addr_valid_pcomp = LogicArray('ldq_addr_valid_pcomp', 'r', configs.numLdqEntries)
            stq_valid_pcomp      = LogicArray('stq_valid_pcomp', 'r', configs.numStqEntries)
            stq_addr_valid_pcomp = LogicArray('stq_addr_valid_pcomp', 'r', configs.numStqEntries)
            stq_data_valid_pcomp = LogicArray('stq_data_valid_pcomp', 'r', configs.numStqEntries)
            addr_valid_pcomp     = LogicVecArray('addr_valid_pcomp', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same_pcomp      = LogicVecArray('addr_same_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            store_is_older_pcomp = LogicVecArray('store_is_older_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            
            ldq_valid_pcomp.regInit(init=[0]*configs.numLdqEntries)
            ldq_addr_valid_pcomp.regInit()
            stq_valid_pcomp.regInit(init=[0]*configs.numStqEntries)
            stq_addr_valid_pcomp.regInit()
            stq_data_valid_pcomp.regInit()
            addr_same_pcomp.regInit()
            store_is_older_pcomp.regInit()

            for i in range(0, configs.numLdqEntries):
                arch += Op((ldq_valid_pcomp, i), (ldq_valid, i))
                arch += Op((ldq_addr_valid_pcomp, i), (ldq_addr_valid, i))
            for j in range(0, configs.numStqEntries):
                arch += Op((stq_valid_pcomp, j), (stq_valid, j))
                arch += Op((stq_addr_valid_pcomp, j), (stq_addr_valid, j))
                arch += Op((stq_data_valid_pcomp, j), (stq_data_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((store_is_older_pcomp, i, j), (store_is_older, i, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_valid_pcomp, i, j), (ldq_addr_valid_pcomp, i), 'and', (stq_addr_valid_pcomp, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_same_pcomp, i, j), '\'1\'', 'when', (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (ld_st_conflict, i, j),
                        (stq_valid_pcomp, j),         'and',
                        (store_is_older_pcomp, i, j), 'and',
                        '(', (addr_same_pcomp, i, j), 'or', 'not', (stq_addr_valid_pcomp, j), ')'
                    )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass, i, j),
                        (ldq_valid_pcomp, i),        'and',
                        'not', (ldq_issue, i),       'and',
                        (stq_data_valid_pcomp, j),   'and',
                        (addr_same_pcomp, i, j),     'and',
                        (addr_valid_pcomp, i, j)
                    )

            # Load

            load_conflict    = LogicArray('load_conflict', 'w', configs.numLdqEntries)
            load_req_valid   = LogicArray('load_req_valid', 'w', configs.numLdqEntries)
            can_load         = LogicArray('can_load', 'w', configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(load_req_valid[i], ldq_valid_pcomp[i], 'and', 'not', ldq_issue[i], 'and', ldq_addr_valid_pcomp[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load[i], 'not', load_conflict[i], 'and', load_req_valid[i])

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(load_idx_oh[w], can_load_list[w], ldq_head_oh)
                arch += Reduce(load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(can_load_list[w+1][i], 'not', load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store

            st_ld_conflict   = LogicVec('st_ld_conflict', 'w', configs.numLdqEntries)
            store_conflict   = Logic('store_conflict', 'w')
            store_valid      = Logic('store_valid', 'w')
            store_data_valid = Logic('store_data_valid', 'w')
            store_addr_valid = Logic('store_addr_valid', 'w')

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict, i),
                    (ldq_valid_pcomp, i), 'and',
                    'not', MuxIndex(store_is_older_pcomp[i], stq_issue), 'and',
                    '(', MuxIndex(addr_same_pcomp[i], stq_issue), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                )
            # The store conflicts with any load
            arch += Reduce(store_conflict, st_ld_conflict, 'or')
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(store_valid, stq_valid_pcomp, stq_issue)
            arch += MuxLookUp(store_data_valid, stq_data_valid_pcomp, stq_issue)
            arch += MuxLookUp(store_addr_valid, stq_addr_valid_pcomp, stq_issue)
            arch += Op(store_en,
                'not', store_conflict, 'and',
                store_valid, 'and',
                store_data_valid, 'and',
                store_addr_valid
            )
            arch += Op(store_idx, stq_issue)

            stq_last_oh = LogicVec('stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(bypass_en_vec, bypass_idx_oh[i], 'and', can_bypass[i])
                arch += Reduce(bypass_en[i], bypass_en_vec, 'or')
        else:
            addr_valid     = LogicVecArray('addr_valid', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same      = LogicVecArray('addr_same', 'w', configs.numLdqEntries, configs.numStqEntries)

            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_valid, i, j), (ldq_addr_valid, i), 'and', (stq_addr_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op((addr_same, i, j), '\'1\'', 'when', (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (ld_st_conflict, i, j),
                        (stq_valid, j),         'and',
                        (store_is_older, i, j), 'and',
                        '(', (addr_same, i, j), 'or', 'not', (stq_addr_valid, j), ')'
                    )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(
                        (can_bypass, i, j),
                        (ldq_valid, i),        'and',
                        'not', (ldq_issue, i), 'and',
                        (stq_data_valid, j),   'and',
                        (addr_same, i, j),     'and',
                        (addr_valid, i, j)
                    )

            # Load

            load_conflict    = LogicArray('load_conflict', 'w', configs.numLdqEntries)
            load_req_valid   = LogicArray('load_req_valid', 'w', configs.numLdqEntries)
            can_load         = LogicArray('can_load', 'w', configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(load_req_valid[i], ldq_valid[i], 'and', 'not', ldq_issue[i], 'and', ldq_addr_valid[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(can_load[i], 'not', load_conflict[i], 'and', load_req_valid[i])

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(load_idx_oh[w], can_load_list[w], ldq_head_oh)
                arch += Reduce(load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(can_load_list[w+1][i], 'not', load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])
            # Store

            st_ld_conflict   = LogicVec('st_ld_conflict', 'w', configs.numLdqEntries)
            store_conflict   = Logic('store_conflict', 'w')
            store_valid      = Logic('store_valid', 'w')
            store_data_valid = Logic('store_data_valid', 'w')
            store_addr_valid = Logic('store_addr_valid', 'w')

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(
                    (st_ld_conflict, i),
                    (ldq_valid, i), 'and',
                    'not', MuxIndex(store_is_older[i], stq_issue), 'and',
                    '(', MuxIndex(addr_same[i], stq_issue), 'or', 'not', (ldq_addr_valid, i), ')'
                )
            # The store conflicts with any load
            arch += Reduce(store_conflict, st_ld_conflict, 'or')
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(store_valid, stq_valid, stq_issue)
            arch += MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
            arch += MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
            arch += Op(store_en,
                'not', store_conflict, 'and',
                store_valid, 'and',
                store_data_valid, 'and',
                store_addr_valid
            )
            arch += Op(store_idx, stq_issue)

            stq_last_oh = LogicVec('stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(bypass_en_vec, bypass_idx_oh[i], 'and', can_bypass[i])
                arch += Reduce(bypass_en[i], bypass_en_vec, 'or')

    if configs.pipe1:
        # Pipeline Stage 1
        load_idx_oh_p1   = LogicVecArray('load_idx_oh_p1', 'r', configs.numLdMem, configs.numLdqEntries)
        load_en_p1       = LogicArray('load_en_p1', 'r', configs.numLdMem)

        load_hs         = LogicArray('load_hs', 'w', configs.numLdMem)
        load_p1_ready    = LogicArray('load_p1_ready', 'w', configs.numLdMem)

        store_idx_p1     = LogicVec('store_idx_p1', 'r', configs.stqAddrW)
        store_en_p1      = Logic('store_en_p1', 'r')

        store_hs        = Logic('store_hs', 'w')
        store_p1_ready   = Logic('store_p1_ready', 'w')

        bypass_idx_oh_p1 = LogicVecArray('bypass_idx_oh_p1', 'r', configs.numLdqEntries, configs.numStqEntries)
        bypass_en_p1     = LogicArray('bypass_en_p1', 'r', configs.numLdqEntries)

        load_idx_oh_p1.regInit(enable=load_p1_ready)
        load_en_p1.regInit(init=[0]*configs.numLdMem, enable=load_p1_ready)

        store_idx_p1.regInit(enable=store_p1_ready)
        store_en_p1.regInit(init=0, enable=store_p1_ready)

        bypass_idx_oh_p1.regInit()
        bypass_en_p1.regInit(init=[0]*configs.numLdqEntries)

        for w in range(0, configs.numLdMem):
            arch += Op(load_hs[w], load_en_p1[w], 'and', rreq_ready_i[w])
            arch += Op(load_p1_ready[w], load_hs[w], 'or', 'not', load_en_p1[w])

        for w in range(0, configs.numLdMem):
            arch += Op(load_idx_oh_p1[w], load_idx_oh[w])
            arch += Op(load_en_p1[w], load_en[w])

        arch += Op(store_hs, store_en_p1, 'and', wreq_ready_i[0])
        arch += Op(store_p1_ready, store_hs, 'or', 'not', store_en_p1)

        arch += Op(store_idx_p1, store_idx)
        arch += Op(store_en_p1, store_en)

        if configs.pipe0:
            for i in range(0, configs.numLdqEntries):
                arch += Op(bypass_idx_oh_p1[i], bypass_idx_oh_p0[i])
        else:
            for i in range(0, configs.numLdqEntries):
                arch += Op(bypass_idx_oh_p1[i], bypass_idx_oh[i])

        for i in range(0, configs.numLdqEntries):
            arch += Op(bypass_en_p1[i], bypass_en[i])

        ######    Read/Write    ######
        # Read Request
        for w in range(0, configs.numLdMem):
            arch += Op(rreq_valid_o[w], load_en_p1[w])
            arch += OHToBits(rreq_id_o[w], load_idx_oh_p1[w])
            arch += Mux1H(rreq_addr_o[w], ldq_addr, load_idx_oh_p1[w])

        for i in range(0, configs.numLdqEntries):
            ldq_issue_set_vec = LogicVec(f'ldq_issue_set_vec_{i}', 'w', configs.numLdMem)
            for w in range(0, configs.numLdMem):
                arch += Op((ldq_issue_set_vec, w),
                    '(', (load_idx_oh, w, i), 'and',
                    (load_p1_ready, w), ')', 'or',
                    (bypass_en, i)
                )
            arch += Reduce(ldq_issue_set[i], ldq_issue_set_vec, 'or')

        # Write Request
        arch += Op(wreq_valid_o[0], store_en_p1)
        arch += Op(wreq_id_o[0], 0)
        arch += MuxLookUp(wreq_addr_o[0], stq_addr, store_idx_p1)
        arch += MuxLookUp(wreq_data_o[0], stq_data, store_idx_p1)
        arch += Op(stq_issue_en, store_en, 'and', store_p1_ready)

        # Read Response and Bypass
        for i in range(0, configs.numLdqEntries):
            # check each read response channel for each load
            read_idx_oh = LogicArray(f'read_idx_oh_{i}', 'w', configs.numLdMem)
            read_valid  = Logic(f'read_valid_{i}', 'w')
            read_data   = LogicVec(f'read_data_{i}', 'w', configs.dataW)
            for w in range(0, configs.numLdMem):
                arch += Op(read_idx_oh[w], rresp_valid_i[w], 'when', '(', rresp_id_i[w], '=', (i, configs.idW), ')', 'else', '\'0\'')
            arch += Mux1H(read_data, rresp_data_i, read_idx_oh)
            arch += Reduce(read_valid, read_idx_oh, 'or')
            # multiplex from store queue data
            bypass_data = LogicVec(f'bypass_data_{i}', 'w', configs.dataW)
            arch += Mux1H(bypass_data, stq_data, bypass_idx_oh_p1[i])
            # multiplex from read and bypass data
            arch += Op(ldq_data[i], read_data, 'or', bypass_data)
            arch += Op(ldq_data_wen[i], bypass_en_p1[i], 'or', read_valid)
        for w in range(0, configs.numLdMem):
            arch += Op(rresp_ready_o[w], '\'1\'')

        # Write Response
        if configs.stResp:
            for i in range(0, configs.numStqEntries):
                arch += Op(stq_exec_set[i],
                    wresp_valid_i[0], 'when',
                    '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                    'else', '\'0\''
                )
        else:
            for i in range(0, configs.numStqEntries):
                arch += Op(stq_reset[i],
                    wresp_valid_i[0], 'when',
                    '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                    'else', '\'0\''
                )
        arch += Op(stq_resp_en, wresp_valid_i[0])
        arch += Op(wresp_ready_o[0], '\'1\'')
    else:
        ######    Read/Write    ######
        # Read Request
        for w in range(0, configs.numLdMem):
            arch += Op(rreq_valid_o[w], load_en[w])
            arch += OHToBits(rreq_id_o[w], load_idx_oh[w])
            arch += Mux1H(rreq_addr_o[w], ldq_addr, load_idx_oh[w])

        for i in range(0, configs.numLdqEntries):
            ldq_issue_set_vec = LogicVec(f'ldq_issue_set_vec_{i}', 'w', configs.numLdMem)
            for w in range(0, configs.numLdMem):
                arch += Op((ldq_issue_set_vec, w),
                    '(', (load_idx_oh, w, i), 'and',
                    (rreq_ready_i, w), 'and',
                    (load_en, w), ')', 'or',
                    (bypass_en, i)
                )
            arch += Reduce(ldq_issue_set[i], ldq_issue_set_vec, 'or')

        # Write Request
        arch += Op(wreq_valid_o[0], store_en)
        arch += Op(wreq_id_o[0], 0)
        arch += MuxLookUp(wreq_addr_o[0], stq_addr, store_idx)
        arch += MuxLookUp(wreq_data_o[0], stq_data, store_idx)
        arch += Op(stq_issue_en, store_en, 'and', wreq_ready_i[0])

        # Read Response and Bypass
        for i in range(0, configs.numLdqEntries):
            # check each read response channel for each load
            read_idx_oh = LogicArray(f'read_idx_oh_{i}', 'w', configs.numLdMem)
            read_valid  = Logic(f'read_valid_{i}', 'w')
            read_data   = LogicVec(f'read_data_{i}', 'w', configs.dataW)
            for w in range(0, configs.numLdMem):
                arch += Op(read_idx_oh[w], rresp_valid_i[w], 'when', '(', rresp_id_i[w], '=', (i, configs.idW), ')', 'else', '\'0\'')
            arch += Mux1H(read_data, rresp_data_i, read_idx_oh)
            arch += Reduce(read_valid, read_idx_oh, 'or')
            # multiplex from store queue data
            bypass_data = LogicVec(f'bypass_data_{i}', 'w', configs.dataW)
            if configs.pipe0:
                arch += Mux1H(bypass_data, stq_data, bypass_idx_oh_p0[i])
            else:
                arch += Mux1H(bypass_data, stq_data, bypass_idx_oh[i])
            # multiplex from read and bypass data
            arch += Op(ldq_data[i], read_data, 'or', bypass_data)
            arch += Op(ldq_data_wen[i], bypass_en[i], 'or', read_valid)
        for w in range(0, configs.numLdMem):
            arch += Op(rresp_ready_o[w], '\'1\'')

        # Write Response
        if configs.stResp:
            for i in range(0, configs.numStqEntries):
                arch += Op(stq_exec_set[i],
                    wresp_valid_i[0], 'when',
                    '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                    'else', '\'0\''
                )
        else:
            for i in range(0, configs.numStqEntries):
                arch += Op(stq_reset[i],
                    wresp_valid_i[0], 'when',
                    '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                    'else', '\'0\''
                )
        arch += Op(stq_resp_en, wresp_valid_i[0])
        arch += Op(wresp_ready_o[0], '\'1\'')


    ######   Write To File  ######
    portInitString += '\n\t);'
    regInitString  += '\tend process;\n'

    # Write to the file
    with open(f'{path_rtl}/{name}.vhd', 'a') as file:
    # with open(name + '.vhd', 'w') as file:
        file.write(library)
        file.write(f'entity {name} is\n')
        file.write(portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name} is\n')
        file.write(signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write(regInitString + 'end architecture;\n')

#===----------------------------------------------------------------------===#
# Final Module Generation
#===----------------------------------------------------------------------===#

def codeGen(path_rtl, configs):
    name = configs.name
    # empty the file
    file = open(f'{path_rtl}/{name}_core.vhd', 'w').close()
    # Group Allocator
    GroupAllocator(path_rtl, name, '_core_ga', configs)
    # Load Address Port Dispatcher
    PortToQueueDispatcher(path_rtl, name, '_core_lda',
        configs.numLdPorts, configs.numLdqEntries, configs.addrW, configs.ldpAddrW
    )
    # Load Data Port Dispatcher
    QueueToPortDispatcher(path_rtl, name, '_core_ldd',
        configs.numLdPorts, configs.numLdqEntries, configs.dataW, configs.ldpAddrW
    )
    # Store Address Port Dispatcher
    PortToQueueDispatcher(path_rtl, name, '_core_sta',
        configs.numStPorts, configs.numStqEntries, configs.addrW, configs.stpAddrW
    )
    # Store Data Port Dispatcher
    PortToQueueDispatcher(path_rtl, name, '_core_std',
        configs.numStPorts, configs.numStqEntries, configs.dataW, configs.stpAddrW
    )
    # Store Backward Port Dispatcher
    if configs.stResp:
        QueueToPortDispatcher(path_rtl, name, '_core_stb',
            configs.numStPorts, configs.numStqEntries, 0, configs.stpAddrW
        )

    # Change the name of the following module to lsq_core
    LSQ(path_rtl, name + '_core', configs)

if __name__ == '__main__':
    # Parse the arguments
    # python3 main.py [-h] [--target-dir PATH_RTL] --spec-file PATH_CONFIGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', '-t', dest='path_rtl', default = './', type = str)
    parser.add_argument('--spec-file', '-s', required = True, dest='path_configs', default = '', type = str)
    args = parser.parse_args()
    path_configs = args.path_configs
    path_rtl     = args.path_rtl

    # Read the configuration file
    lsqConfig = GetConfigs(path_configs)
    codeGen(path_rtl, lsqConfig)

