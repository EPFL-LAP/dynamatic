# ===----------------------------------------------------------------------===#
# Signal Definition
# ===----------------------------------------------------------------------===#
# This section defined Python classes that generate signal declarations.
#
# - class Logic         : (std_logic) one‑bit signal wire / port / register
# - class LogicVec      : (std_logic_vector) Multi-bit signal.
# - class LogicArray    : (Multiple std_logic) Array of individual std_logic signals.
# - class LogicVecArray : (Multiple std_logic_vector) Array of std_logic_vector signals.

#
# std_logic bit
#

from core_gen.ir import Statement


class Logic(Statement):
    """
    A one-bit std_logic signal.

    Logic class encapsulates wires, ports, and registers in the code generator,
    handling name with '_i', '_o', '_q', '_d' suffixes.

    Attributes:
        em (Emitter): Emitter for code generation.
        name (str): The base name of the signal.
        type (str):
            'i' input port      (<name>_i: in std_logic)
            'o' output port     (<name>_o: out std_logic)
            'w' internal wire   (signal <name>: std_logic)
            'r' register        (<name>_q) for the registered value
                                (<name>_d) for the next-cycled value
        custom_suffix (str): Replaces the default '_i'/'_o' port suffix, see __init__
        force_reg (bool): Flag indicating whether to force the signal to be a register, only relevant for Verilog generation

    Methods:
        getBaseNameRead(): Returns the read name of the (sufix-indexed) signal.
        getBaseNameWrite(): Returns the write name of the (sufix-indexed) signal.
        getNameRead(): Returns the name we should use when reading the signal. (e.g. <name>_q for a register type)
        getNameWrite(): Returns the name to write to. (e.g. <name>_d for a register type)
        signalInit(): Appends the signal/port declaration.
        regInit(): Appends the register initialization block.
    """

    # Signal name
    name = ''
    # Signal type, 'i' for input, 'o' for output, 'w' for wire, 'r' for register
    type = ''

    def __init__(
        self,
        em: Statement,
        name: str,
        type: str = 'w',
        init: bool = True,
        custom_suffix=None,
        force_reg=False,
    ) -> None:
        """
        init: If True, immediately generates the corresponding std_logic in VHDL.
              True when we instantiate Logic.
              False when we instantiate LogicVec, LogicArray, and LogicVecArray.
        custom_suffix: If given, appended to the name (after any array index)
              instead of the default '_i'/'_o' port suffix,
              e.g. name 'io_ctrl' with custom_suffix '_ready' gives 'io_ctrl_0_ready'.
              Not allowed for registers, which need '_d'/'_q' to be distinguishable.
              This is a workaround for the Dynamatic handshake naming convention
              (<name>_<index>_valid/ready), currently only used by the LSQ wrapper.
        """
        # Type should be one of the four types.
        assert type in ('i', 'o', 'w', 'r')
        assert (
            type != 'r' or custom_suffix is None
        ), f'Register "{name}" cannot have a custom suffix!'
        self.em = em
        self.name = name
        self.type = type
        self.custom_suffix = custom_suffix
        if custom_suffix is not None:
            self.suffix = custom_suffix
        elif type == 'i':
            self.suffix = '_i'
        elif type == 'o':
            self.suffix = '_o'
        else:
            self.suffix = ''
        self.force_reg = force_reg
        if init:
            self.signalInit()

    def __repr__(self) -> str:
        """
        Print Logic with useful information.
        """
        # Signal type
        type = ''
        if self.type == 'w':
            type = 'wire'
        elif self.type == 'i':
            type = 'input'
        elif self.type == 'o':
            type = 'output'
        elif self.type == 'r':
            type = 'reg'
        return (
            f'name: {self.name}\n'
            + f'type: {type}\n'
            + f'size: single bit\n'
        )

    def getBaseNameRead(self, sufix='') -> str:
        """
        Returns the read name of the (sufix-indexed) signal, e.g. <name><sufix>_q for a register.
        Not overridden, so subclasses use it to build their indexed names.
        """
        if self.type == 'o':
            raise TypeError(f'Cannot read from the output signal "{self.name}"!')
        name = self.name + sufix + self.suffix
        if self.type == 'r':
            return name + '_q'
        return name

    def getBaseNameWrite(self, sufix='') -> str:
        """
        Returns the write name of the (sufix-indexed) signal, e.g. <name><sufix>_d for a register.
        """
        if self.type == 'i':
            raise TypeError(f'Cannot write to the input signal "{self.name}"!')
        name = self.name + sufix + self.suffix
        if self.type == 'r':
            return name + '_d'
        return name

    def getNameRead(self) -> str:
        """
        Returns the name we should use when reading the signal.

        Example (Pseudo-code)
            If you want to do "Logic a = Logic b + Logic c"
            -> getNameWrite(a) = getNameRead(b) + getNameRead(c)
        """
        return self.getBaseNameRead()

    def _to_str(self, em: Statement, size) -> str:
        return self.getNameRead()

    def getNameWrite(self) -> str:
        """
        Returns the name to write to.

        Example in the getNameRead() method.
        """
        return self.getBaseNameWrite()

    def signalInit(self, sufix='') -> None:
        """
        Appends the appropriate declaration or port line for this signal to a global buffer.
        """
        self.em.logic_signal_init(self, sufix)

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
        self.em.logic_reg_init(self, enable, init)


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

    def __init__(
        self,
        em: Statement,
        name: str,
        type: str = 'w',
        size: int = 1,
        init: bool = True,
        custom_suffix=None,
        force_reg=False,
    ) -> None:
        Logic.__init__(self, em, name, type, False, custom_suffix, force_reg)
        assert size > 0
        self.size = size
        if init:
            self.signalInit()

    def __repr__(self) -> str:
        # Signal type
        type = ''
        if self.type == 'w':
            type = 'wire'
        elif self.type == 'i':
            type = 'input'
        elif self.type == 'o':
            type = 'output'
        elif self.type == 'r':
            type = 'reg'
        return (
            f'name: {self.name}\n'
            + f'type: {type}\n'
            + f'size: {self.size}\n'
        )

    def getNameRead(self, i=None) -> str:
        if i == None:
            return self.getBaseNameRead()
        else:
            assert i < self.size
            return self.em.index_var(self.getBaseNameRead(), i)

    def getNameWrite(self, i=None) -> str:
        if i == None:
            return self.getBaseNameWrite()
        else:
            assert i < self.size
            return self.em.index_var(self.getBaseNameWrite(), i)

    def signalInit(self, sufix=''):
        self.em.logicvec_signal_init(self, sufix)

    def regInit(self, enable=None, init=None) -> None:
        self.em.logicvec_reg_init(self, enable, init)


#
# An array of std_logic
#


class LogicArray(Logic):
    """
    Represents a N-length array of one-bit std_logic.
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

    def __init__(
        self,
        em: Statement,
        name: str,
        type: str = 'w',
        length: int = 1,
        custom_suffix=None,
        force_reg=False,
    ):
        self.length = length
        Logic.__init__(self, em, name, type, False, custom_suffix, force_reg)
        self.signalInit()

    def __repr__(self) -> str:
        return Logic.__repr__(self) + f'array length: {self.length}'

    def getNameRead(self, i) -> str:
        assert i in range(0, self.length)
        return self.getBaseNameRead(f'_{i}')

    def getNameWrite(self, i) -> str:
        assert i in range(0, self.length)
        return self.getBaseNameWrite(f'_{i}')

    def signalInit(self) -> None:
        for i in range(0, self.length):
            Logic.signalInit(self, f'_{i}')

    def __getitem__(self, i) -> Logic:
        assert i in range(0, self.length)
        return Logic(
            self.em, self.name + f'_{i}', self.type, False, self.custom_suffix
        )

    def regInit(self, enable=None, init=None) -> None:
        self.em.logicarray_reg_init(self, enable, init)


#
# An array of std_logic vector
#


class LogicVecArray(LogicVec):
    """
    Represents a N-length array of M-bit std_logic_vec.
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

    def __init__(
        self,
        em: Statement,
        name: str,
        type: str = 'w',
        length: int = 1,
        size: int = 1,
        custom_suffix=None,
        force_reg=False,
    ):
        self.length = length
        LogicVec.__init__(self, em, name, type, size, False, custom_suffix, force_reg)
        self.signalInit()

    def __repr__(self) -> str:
        return LogicVec.__repr__(self) + f'array length: {self.length}'

    def getNameRead(self, i, j=None) -> str:
        assert i in range(0, self.length)
        if j == None:
            return self.getBaseNameRead(f'_{i}')
        assert j < self.size
        return self.em.index_var(self.getBaseNameRead(f'_{i}'), j)

    def getNameWrite(self, i, j=None) -> str:
        assert i in range(0, self.length)
        if j == None:
            return self.getBaseNameWrite(f'_{i}')
        assert j < self.size
        return self.em.index_var(self.getBaseNameWrite(f'_{i}'), j)

    def signalInit(self) -> None:
        for i in range(0, self.length):
            LogicVec.signalInit(self, f'_{i}')

    def __getitem__(self, i) -> LogicVec:
        assert i in range(0, self.length)
        return LogicVec(
            self.em,
            self.name + f'_{i}',
            self.type,
            self.size,
            False,
            self.custom_suffix,
        )

    def regInit(self, enable=None, init=None) -> None:
        self.em.logicvecarray_reg_init(self, enable, init)
