# ===----------------------------------------------------------------------===#
# VHDL Signal Definition
# ===----------------------------------------------------------------------===#
# This section defined Python classes that generate VHDL signal declarations.
#
# - class Logic         : (std_logic) one‑bit signal wire / port / register
# - class LogicVec      : (std_logic_vector) Multi-bit signal.
# - class LogicArray    : (Multiple std_logic) Array of individual std_logic signals.
# - class LogicVecArray : (Multiple std_logic_vector) Array of std_logic_vector signals.

#
# std_logic bit
#

from vhdl_gen.context import VHDLContext
from vhdl_gen.utils import *


class Logic:
    """
    A one-bit VHDL std_logic signal.

    Logic class encapsulates wires, ports, and registers in the code generator,
    handling name with '_i', '_o', '_q', '_d' suffixes.

    Attributes:
        ctx (VHDLContext): Context for code generation.
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

    def __init__(self, ctx: VHDLContext, name: str, type: str = 'w', init: bool = True) -> None:
        """
        init: If True, immediately generates the corresponding std_logic in VHDL.
              True when we instantiate Logic.
              False when we instantiate LogicVec, LogicArray, and LogicVecArray.
        """
        # Type should be one of the four types.
        assert (type in ('i', 'o', 'w', 'r'))
        self.ctx = ctx
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
        if (self.type == 'w'):
            type = 'wire'
        elif (self.type == 'i'):
            type = 'input'
        elif (self.type == 'o'):
            type = 'output'
        elif (self.type == 'r'):
            type = 'reg'
        return f'name: {self.name}\n' + f'type: {type}\n' + f'size: single bit\n'

    def getNameRead(self, sufix='') -> str:
        """
        Returns the name we should use when reading the signal.

        Example (Pseudo-code)
            If you want to do "Logic a = Logic b + Logic c"
            -> getNameWrite(a) = getNameRead(b) + getNameRead(c)
        """
        if (self.type == 'w'):
            return self.name + sufix
        elif (self.type == 'r'):
            return self.name + sufix + '_q'
        elif (self.type == 'i'):
            return self.name + sufix + '_i'
        elif (self.type == 'o'):
            raise TypeError(f'Cannot read from the output signal \"{self.name}\"!')

    def getNameWrite(self, sufix='') -> str:
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

    def signalInit(self, sufix='') -> None:
        """
        Appends the appropriate declaration or port line for this signal to a global buffer.
        """
        if (self.type == 'w'):
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix} : std_logic;\n')
        elif (self.type == 'r'):
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix}_d : std_logic;\n')
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix}_q : std_logic;\n')
        elif (self.type == 'i'):
            self.ctx.add_port_str(';\n')
            self.ctx.add_port_str(f'\t\t{self.name + sufix}_i : in std_logic')
        elif (self.type == 'o'):
            self.ctx.add_port_str(';\n')
            self.ctx.add_port_str(f'\t\t{self.name + sufix}_o : out std_logic')

    def regInit(self, enable=None, init=None) -> None:
        """
        Generates a clocked process snippet that sets up the register's behavior.
        For example,

        if (rst = '1') then
            <name>_q <= '0';
        elsif (rising_edge(clk)) then
            <name>_q <= <name>_d;
        end if;
        """
        assert (self.type == 'r')
        if (init != None):
            self.ctx.add_reg_str('\t\tif (rst = \'1\') then\n')
            self.ctx.add_reg_str(f'\t\t\t{self.getNameRead()} <= {IntToBits(init)};\n')
            self.ctx.add_reg_str('\t\telsif (rising_edge(clk)) then\n')
        else:
            self.ctx.add_reg_str('\t\tif (rising_edge(clk)) then\n')
        if (enable != None):
            self.ctx.add_reg_str(f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n')
            self.ctx.add_reg_str(f'\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n')
            self.ctx.add_reg_str('\t\t\tend if;\n')
        else:
            self.ctx.add_reg_str(f'\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n')
        self.ctx.add_reg_str('\t\tend if;\n')

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

    def __init__(self, ctx: VHDLContext, name: str, type: str = 'w', size: int = 1, init: bool = True) -> None:
        Logic.__init__(self, ctx, name, type, False)
        assert (size > 0)
        self.size = size
        if (init):
            self.signalInit()

    def __repr__(self) -> str:
        # Signal type
        type = ''
        if (self.type == 'w'):
            type = 'wire'
        elif (self.type == 'i'):
            type = 'input'
        elif (self.type == 'o'):
            type = 'output'
        elif (self.type == 'r'):
            type = 'reg'
        return f'name: {self.name}\n' + f'type: {type}\n' + f'size: {self.size}\n'

    def getNameRead(self, i=None, sufix='') -> str:
        if (i == None):
            return Logic.getNameRead(self, sufix)
        else:
            assert (i < self.size)
            return Logic.getNameRead(self, sufix) + f'({i})'

    def getNameWrite(self, i=None, sufix='') -> str:
        if (i == None):
            return Logic.getNameWrite(self, sufix)
        else:
            assert (i < self.size)
            return Logic.getNameWrite(self, sufix) + f'({i})'

    def signalInit(self, sufix=''):
        if (self.type == 'w'):
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix} : std_logic_vector({self.size-1} downto 0);\n')
        elif (self.type == 'r'):
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix}_d : std_logic_vector({self.size-1} downto 0);\n')
            self.ctx.add_signal_str(f'\tsignal {self.name + sufix}_q : std_logic_vector({self.size-1} downto 0);\n')
        elif (self.type == 'i'):
            self.ctx.add_port_str(';\n')
            self.ctx.add_port_str(f'\t\t{self.name + sufix}_i : in std_logic_vector({self.size-1} downto 0)')
        elif (self.type == 'o'):
            self.ctx.add_port_str(';\n')
            self.ctx.add_port_str(f'\t\t{self.name + sufix}_o : out std_logic_vector({self.size-1} downto 0)')

    def regInit(self, enable=None, init=None) -> None:
        assert (self.type == 'r')
        if (init != None):
            self.ctx.add_reg_str('\t\tif (rst = \'1\') then\n')
            self.ctx.add_reg_str(f'\t\t\t{self.getNameRead()} <= {IntToBits(init, self.size)};\n')
            self.ctx.add_reg_str('\t\telsif (rising_edge(clk)) then\n')
        else:
            self.ctx.add_reg_str('\t\tif (rising_edge(clk)) then\n')
        if (enable != None):
            self.ctx.add_reg_str(f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n')
            self.ctx.add_reg_str(f'\t\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n')
            self.ctx.add_reg_str('\t\t\tend if;\n')
        else:
            self.ctx.add_reg_str(f'\t\t\t{self.getNameRead()} <= {self.getNameWrite()};\n')
        self.ctx.add_reg_str('\t\tend if;\n')

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

    def __init__(self, ctx: VHDLContext, name: str, type: str = 'w', length: int = 1):
        self.length = length
        Logic.__init__(self, ctx, name, type, False)
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
        return Logic(self.ctx, self.name + f'_{i}', self.type, False)

    def regInit(self, enable=None, init=None) -> None:
        assert (self.type == 'r')
        if (init != None):
            self.ctx.add_reg_str('\t\tif (rst = \'1\') then\n')
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i])};\n')
            self.ctx.add_reg_str('\t\telsif (rising_edge(clk)) then\n')
        else:
            self.ctx.add_reg_str('\t\tif (rising_edge(clk)) then\n')
        if (enable != None):
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n')
                self.ctx.add_reg_str(f'\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n')
                self.ctx.add_reg_str('\t\t\tend if;\n')
        else:
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n')
        self.ctx.add_reg_str('\t\tend if;\n')

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

    def __init__(self, ctx: VHDLContext, name: str, type: str = 'w', length: int = 1, size: int = 1):
        self.length = length
        LogicVec.__init__(self, ctx, name, type, size, False)
        self.signalInit()

    def __repr__(self) -> str:
        return LogicVec.__repr__(self) + f'array length: {self.length}'

    def getNameRead(self, i, j=None) -> str:
        assert i in range(0, self.length)
        return LogicVec.getNameRead(self, j, f'_{i}')

    def getNameWrite(self, i, j=None) -> str:
        assert i in range(0, self.length)
        return LogicVec.getNameWrite(self, j, f'_{i}')

    def signalInit(self) -> None:
        for i in range(0, self.length):
            LogicVec.signalInit(self, f'_{i}')

    def __getitem__(self, i) -> LogicVec:
        assert i in range(0, self.length)
        return LogicVec(self.ctx, self.name + f'_{i}', self.type, self.size, False)

    def regInit(self, enable=None, init=None) -> None:
        assert (self.type == 'r')
        if (init != None):
            self.ctx.add_reg_str('\t\tif (rst = \'1\') then\n')
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\t{self.getNameRead(i)} <= {IntToBits(init[i], self.size)};\n')
            self.ctx.add_reg_str('\t\telsif (rising_edge(clk)) then\n')
        else:
            self.ctx.add_reg_str('\t\tif (rising_edge(clk)) then\n')
        if (enable != None):
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n')
                self.ctx.add_reg_str(f'\t\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n')
                self.ctx.add_reg_str('\t\t\tend if;\n')
        else:
            for i in range(0, self.length):
                self.ctx.add_reg_str(f'\t\t\t{self.getNameRead(i)} <= {self.getNameWrite(i)};\n')
        self.ctx.add_reg_str('\t\tend if;\n')
