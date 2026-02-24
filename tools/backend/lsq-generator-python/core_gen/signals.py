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

from core_gen.utils import *
from core_gen.ir import Statement


class Logic(Statement):
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

    def __init__(self, ctx: Statement, name: str, type: str = 'w', init: bool = True, dyn_comp=False, force_reg=False) -> None:
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
        self.dyn_comp = dyn_comp
        self.force_reg = force_reg
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
        return f'name: {self.get_base_name()}\n' + f'type: {type}\n' + f'size: single bit\n'

    def getNameRead(self, sufix='') -> str:
        """
        Returns the name we should use when reading the signal.

        Example (Pseudo-code)
            If you want to do "Logic a = Logic b + Logic c"
            -> getNameWrite(a) = getNameRead(b) + getNameRead(c)
        """
        if (self.type == 'w'):
            return self.get_base_name(sufix)
        elif (self.type == 'r'):
            return self.get_base_name(sufix) + '_q'
        elif (self.type == 'i'):
            return self.get_base_name(sufix) + ('_i' if not self.dyn_comp else '')
        elif (self.type == 'o'):
            raise TypeError(f'Cannot read from the output signal \"{self.get_base_name(sufix)}\"!')

    def _to_str(self, em: Statement, size) -> str:
        return self.getNameRead()

    def getNameWrite(self, sufix='') -> str:
        """
        Returns the name to write to. 

        Example in the getNameRead() method.
        """
        if (self.type == 'w'):
            return self.get_base_name(sufix)
        elif (self.type == 'r'):
            return self.get_base_name(sufix) + '_d'
        elif (self.type == 'i'):
            raise TypeError(f'Cannot write to the input signal \"{self.get_base_name(sufix)}\"!')
        elif (self.type == 'o'):
            return self.get_base_name(sufix) + ('_o' if not self.dyn_comp else '')

    def signalInit(self, sufix='') -> None:
        self.ctx.logic_signal_init(self, sufix)

    def regInit(self, enable=None, init=None) -> None:
        self.ctx.logic_reg_init(self, enable, init)

    def get_base_name(self, sufix='') -> str:
        if not self.dyn_comp or sufix == '':
            return self.name + sufix
        
        name_list = self.name.split('_')
        name_list = name_list[:-1] + [sufix, name_list[-1]]
        name = '_'.join(name_list)
        return name.replace('__', '_')
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

    def __init__(self, ctx: Statement, name: str, type: str = 'w', size: int = 1, init: bool = True, dyn_comp=False, force_reg=False) -> None:
        Logic.__init__(self, ctx, name, type, False, dyn_comp, force_reg)
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
        return f'name: {self.get_base_name()}\n' + f'type: {type}\n' + f'size: {self.size}\n'

    def getNameRead(self, i=None, sufix='') -> str:
        if (i == None):
            return Logic.getNameRead(self, sufix)
        else:
            assert (i < self.size)
            return self.ctx.index_var(Logic.getNameRead(self, sufix), i)

    def getNameWrite(self, i=None, sufix='') -> str:
        if (i == None):
            return Logic.getNameWrite(self, sufix)
        else:
            assert (i < self.size)
            return self.ctx.index_var(Logic.getNameWrite(self, sufix), i)

    def signalInit(self, sufix=''):
        self.ctx.logicvec_signal_init(self, sufix)

    def regInit(self, enable=None, init=None) -> None:
        self.ctx.logicvec_reg_init(self, enable, init)

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

    def __init__(self, ctx: Statement, name: str, type: str = 'w', length: int = 1, dyn_comp=False, force_reg=False):
        self.length = length
        Logic.__init__(self, ctx, name, type, False, dyn_comp, force_reg)
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
        return Logic(self.ctx, self.get_base_name(f'_{i}'), self.type, False, self.dyn_comp)

    def regInit(self, enable=None, init=None) -> None:
        self.ctx.logicarray_reg_init(self, enable, init)
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

    def __init__(self, ctx: Statement, name: str, type: str = 'w', length: int = 1, size: int = 1, dyn_comp=False, force_reg=False):
        self.length = length
        LogicVec.__init__(self, ctx, name, type, size, False, dyn_comp, force_reg)
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
        return LogicVec(self.ctx, self.get_base_name(f'_{i}'), self.type, self.size, False, self.dyn_comp)

    def regInit(self, enable=None, init=None) -> None:
        self.ctx.logicvecarray_reg_init(self, enable, init)