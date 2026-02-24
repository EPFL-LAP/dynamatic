from verilog_gen.emitters.emitter import Emitter, Meta
from verilog_gen.ir import Statement, Bin, Un, BinOp, UnOp, Bit, WhenElse, Type
from verilog_gen.signals import Logic, LogicVec, LogicArray, LogicVecArray
# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class VHDLEmitter(Emitter):
    """
    A context object to replace global variables for VHDL code generation.
    Holds indentation level, temporary name counter, and initialization strings.
    """

    def __init__(self, reset_name='rst', clock_name='clk'):
        # Indentation level for generated code
        self.tabLevel = 1

        # Counter for generating unique temporary names
        self.tempCount = 0

        # Accumulated initialization code sections
        self.signalInitString = ''
        self.reset_name = reset_name
        self.clock_name = clock_name

        self.portInitString = ''
        self.regInitString = ''
        self.statementString = ''

        self.inst_started = False

        # Default library imports for VHDL
        self.library = 'library IEEE;\nuse IEEE.std_logic_1164.all;\nuse IEEE.numeric_std.all;\n\n'
        
    def get_port_init_str(self) -> str:
        return f'port(\n\t\t{self.reset_name} : in std_logic;\n\t\t{self.clock_name} : in std_logic'
    
    def get_port_end_str(self) -> str:
        return f'\n{self.get_current_indent()});'

    def get_reg_init_str(self) -> str:
        return f'process ({self.clock_name}, {self.reset_name}) is\n{self.get_current_indent()}begin\n'
    
    def get_reg_end_str(self) -> str:
        return f'end process;\n'

    def get_current_indent(self) -> str:
        return '\t' * self.tabLevel

    def increase_indent(self):
        self.tabLevel += 1

    def decrease_indent(self):
        self.tabLevel = max(0, self.tabLevel - 1)

    def get_temp(self, name: str) -> str:
        return f'TEMP_{self.tempCount}_{name}'

    def use_temp(self):
        self.tempCount += 1

    def add_signal_str(self, code: str):
        self.signalInitString += code

    def add_port_str(self, code: str):
        self.portInitString += code

    def add_reg_str(self, code: str):
        self.regInitString += code

    def add_statement(self, code: str):
        self.statementString += self.get_current_indent() + code

    def add_comment(self, comment: str):
        self.statementString += self.get_current_indent() + f'-- {comment}\n'

    def add_assignment(self, out, statement: Statement, in_process=False):
        out_str, size = self.assigned_var_to_str(out)
        meta = Meta(size, Type.LOGIC, -1)
        statement_str = statement.to_str(self, meta)
        # Assume we only write to logic types
        statement_str = self.fix_type(Type.LOGIC, statement.get_type(), statement_str)
        self.statementString += self.get_current_indent() + f'{out_str} <= {statement_str};\n'

    def get_definition_str(self, module_name: str, write_regs=True) -> str:
        return self.library + \
                f'entity {module_name} is\n' + \
                self.get_current_indent() + self.get_port_init_str() + self.portInitString + self.get_port_end_str() + \
                '\nend entity;\n\n' + \
                f'architecture arch of {module_name} is\n' + \
                self.signalInitString + \
                'begin\n' + self.statementString + '\n' + \
                ((self.get_current_indent() + self.get_reg_init_str() + self.regInitString + self.get_reg_end_str()) if write_regs and self.regInitString != '' else '') \
                + 'end architecture;\n'
    
    def start_instantiation(self, module_name:str, instance_name: str = None) -> str:
        if self.inst_started: # Sanity check to prevent overlapping instantiations
            raise ValueError('start_instantiation called while another instantiation is in progress')

        if instance_name is None: instance_name = module_name

        self.inst_started = True
        self.first_map = True
        self.inst_str = f'{self.get_current_indent()}{instance_name} : entity work.{module_name}\n'
        self.increase_indent()
        self.inst_str += f'{self.get_current_indent()}port map('
        self.increase_indent()

    def add_map(self, port_name: str, signal_name: str='open') -> str:
        if not self.inst_started:
            raise ValueError('add_map can only be called after start_instantiation')
        
        assert isinstance(port_name, str) and isinstance(signal_name, str), "port name and signal name must be strings"

        if not self.first_map:
            self.inst_str += ','
        else:
            self.first_map = False

        self.inst_str += f'\n{self.get_current_indent()}{port_name} => {signal_name}'

    def complete_instantiation(self) -> str:
        self.inst_started = False
        self.decrease_indent()
        self.inst_str += f'\n{self.get_current_indent()});\n'
        self.decrease_indent()
        self.statementString += self.inst_str
        self.inst_str = ''

    BINOP_STRINGS = {
        BinOp.ADD: '+',
        BinOp.SUB: '-',
        BinOp.AND: 'and',
        BinOp.OR: 'or',
        BinOp.XOR: 'xor',
        BinOp.MUL: '*',
        BinOp.GE: '>=',
        BinOp.LE: '<=',
        BinOp.GT: '>',
        BinOp.LT: '<',
        BinOp.EQ: '=',
        BinOp.NEQ: '!=',
        BinOp.CONCAT: '&'
    }

    @staticmethod
    def is_surrounded_by_parentheses(s: str) -> bool:
        s = s.strip()
        if not s.startswith('(') or not s.endswith(')'):
            return False

        depth = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i == len(s) - 1
                if depth < 0:
                    return False
        return False

    def get_binop_str(self, op: Bin) -> str:
        if op in self.BINOP_STRINGS:
            return self.BINOP_STRINGS[op]
        else:
            raise ValueError('Invalid binary operator: ' + str(op))
            
    def get_unop_str(self, unop: UnOp) -> str:
        if unop == UnOp.NOT:
            return 'not'
        else:
            raise ValueError('Invalid unary operator')

    def get_bit_str(self, bit: Bit) -> str:
        if bit.value == 0:
            return '\'0\''
        elif bit.value == 1:
            return '\'1\''
        else:
            raise ValueError('Invalid bit value')

    def fix_type(self, super_type: Type, child_type: Type, child_str: str) -> str:
        if super_type == Type.ARITH and child_type == Type.LOGIC:
            return f'unsigned{child_str}' if self.is_surrounded_by_parentheses(child_str) else f'unsigned({child_str})'
        elif super_type == Type.LOGIC and child_type == Type.ARITH:
            return f'std_logic_vector{child_str}' if self.is_surrounded_by_parentheses(child_str) else f'std_logic_vector({child_str})'
        else:
            return child_str

    def bin_to_str(self, bin: Bin, meta: Meta) -> str:
        meta = Meta(meta.size, bin.get_param_type(), bin.get_precedence())
        left_str = bin.left.to_str(self, meta)
        right_str = bin.right.to_str(self, meta)

        left_str = self.fix_type(bin.get_param_type(), bin.left.get_type(), left_str)
        right_str = self.fix_type(bin.get_param_type(), bin.right.get_type(), right_str)

        return f'{left_str} {self.get_binop_str(bin.op)} {right_str}'

    def un_to_str(self, un: Un, meta: Meta) -> str:
        meta = Meta(meta.size, un.get_param_type(), un.get_precedence())
        val_str = un.val.to_str(self, meta)
        val_str = self.fix_type(un.get_param_type(), un.val.get_type(), val_str)
        return f'{self.get_unop_str(un.op)} {val_str}'

    def when_else_to_str(self, when_else, meta: Meta) -> str:
        
        # add a linebreak if the "else" statement is a when-else themselves
        if isinstance(when_else.false_statement, WhenElse):
            """
            a bit of a hack to force the inner when-else to not add parenthesis and an enter when chained
            So the resulting when else will look like:
            [val 1] when [cond 1] else
            [val 2] when [cond 2] else
            [val 3] when [cond 3] else 
            ...
            """
            
            enter = f'\n{self.get_current_indent()}\t'
            self_precedence = -1
        else:
            enter = ' '
            self_precedence = 0
            
        meta = Meta(meta.size, when_else.get_type(), when_else.get_precedence())
        true_str = when_else.true_statement.to_str(self, meta)
        false_str = when_else.false_statement.to_str(self, Meta(meta.size, meta.type, self_precedence))
        cond_str = when_else.condition.to_str(self, meta)

        cond_str = self.fix_type(Type.BOOL, when_else.condition.get_type(), cond_str)
            
        return f'{true_str} when {cond_str} else{enter}{false_str}'

    def logic_signal_init(self, signal: Logic, sufix: str):
        """
        Appends the appropriate declaration or port line for this signal to a global buffer.
        """
        if (signal.type == 'w'):
            self.add_signal_str(f'\tsignal {signal.get_base_name(sufix)} : std_logic;\n')
        elif (signal.type == 'r'):
            self.add_signal_str(f'\tsignal {signal.get_base_name(sufix)}_d : std_logic;\n')
            self.add_signal_str(f'\tsignal {signal.get_base_name(sufix)}_q : std_logic;\n')
        elif (signal.type == 'i'):
            self.add_port_str(';\n')
            self.add_port_str(f'\t\t{signal.get_base_name(sufix)}{"_i" if not signal.dyn_comp else ""} : in std_logic')
        elif (signal.type == 'o'):
            self.add_port_str(';\n')
            self.add_port_str(f'\t\t{signal.get_base_name(sufix)}{"_o" if not signal.dyn_comp else ""} : out std_logic')


    def logicvec_signal_init(self, vec: LogicVec, sufix: str):
        if (vec.type == 'w'):
            self.add_signal_str(f'\tsignal {vec.get_base_name(sufix)} : std_logic_vector({vec.size-1} downto 0);\n')
        elif (vec.type == 'r'):
            self.add_signal_str(f'\tsignal {vec.get_base_name(sufix)}_d : std_logic_vector({vec.size-1} downto 0);\n')
            self.add_signal_str(f'\tsignal {vec.get_base_name(sufix)}_q : std_logic_vector({vec.size-1} downto 0);\n')
        elif (vec.type == 'i'):
            self.add_port_str(';\n')
            self.add_port_str(f'\t\t{vec.get_base_name(sufix)}{'_i' if not vec.dyn_comp else ""} : in std_logic_vector({vec.size-1} downto 0)')
        elif (vec.type == 'o'):
            self.add_port_str(';\n')
            self.add_port_str(f'\t\t{vec.get_base_name(sufix)}{'_o' if not vec.dyn_comp else ""} : out std_logic_vector({vec.size-1} downto 0)')

    def logic_reg_init(self, logic: Logic, enable=None, init=None) -> None:
        """
        Generates a clocked process snippet that sets up the register's behavior.
        For example,

        if (rst = '1') then
            <name>_q <= '0';
        elsif (rising_edge(clk)) then
            <name>_q <= <name>_d;
        end if;
        """
        assert (logic.type == 'r')
        if (init != None):
            self.add_reg_str(f'\t\tif ({self.reset_name} = \'1\') then\n')
            self.add_reg_str(f'\t\t\t{logic.getNameRead()} <= {self.in_to_bits(init)};\n')
            self.add_reg_str(f'\t\telsif (rising_edge({self.clock_name})) then\n')
        else:
            self.add_reg_str(f'\t\tif (rising_edge({self.clock_name})) then\n')
        if (enable != None):
            self.add_reg_str(f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n')
            self.add_reg_str(f'\t\t\t\t{logic.getNameRead()} <= {logic.getNameWrite()};\n')
            self.add_reg_str('\t\t\tend if;\n')
        else:
            self.add_reg_str(f'\t\t\t{logic.getNameRead()} <= {logic.getNameWrite()};\n')
        self.add_reg_str('\t\tend if;\n')

    def logicvec_reg_init(self, vec: LogicVec, enable=None, init=None) -> None:
        assert (vec.type == 'r')
        if (init != None):
            self.add_reg_str(f'\t\tif ({self.reset_name} = \'1\') then\n')
            self.add_reg_str(f'\t\t\t{vec.getNameRead()} <= {self.int_to_str(init, vec.size)};\n')
            self.add_reg_str(f'\t\telsif (rising_edge({self.clock_name})) then\n')
        else:
            self.add_reg_str(f'\t\tif (rising_edge({self.clock_name})) then\n')
        if (enable != None):
            self.add_reg_str(f'\t\t\tif ({enable.getNameRead()} = \'1\') then\n')
            self.add_reg_str(f'\t\t\t\t{vec.getNameRead()} <= {vec.getNameWrite()};\n')
            self.add_reg_str('\t\t\tend if;\n')
        else:
            self.add_reg_str(f'\t\t\t{vec.getNameRead()} <= {vec.getNameWrite()};\n')
        self.add_reg_str('\t\tend if;\n')


    def logicarray_reg_init(self, array: LogicArray, enable=None, init=None) -> None:
        assert (array.type == 'r')
        if (init != None):
            self.add_reg_str(f'\t\tif ({self.reset_name} = \'1\') then\n')
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\t{array.getNameRead(i)} <= {self.int_to_str(init[i])};\n')
            self.add_reg_str(f'\t\telsif (rising_edge({self.clock_name})) then\n')
        else:
            self.add_reg_str(f'\t\tif (rising_edge({self.clock_name})) then\n')
        if (enable != None):
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n')
                self.add_reg_str(f'\t\t\t\t{array.getNameRead(i)} <= {array.getNameWrite(i)};\n')
                self.add_reg_str('\t\t\tend if;\n')
        else:
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\t{array.getNameRead(i)} <= {array.getNameWrite(i)};\n')
        self.add_reg_str('\t\tend if;\n')


    def logicvecarray_reg_init(self, array: LogicVecArray, enable=None, init=None) -> None:
        assert (array.type == 'r')
        if (init != None):
            self.add_reg_str(f'\t\tif ({self.reset_name} = \'1\') then\n')
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\t{array.getNameRead(i)} <= {self.int_to_str(init[i], array.size)};\n')
            self.add_reg_str(f'\t\telsif (rising_edge({self.clock_name})) then\n')
        else:
            self.add_reg_str(f'\t\tif (rising_edge({self.clock_name})) then\n')
        if (enable != None):
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\tif ({enable.getNameRead(i)} = \'1\') then\n')
                self.add_reg_str(f'\t\t\t\t{array.getNameRead(i)} <= {array.getNameWrite(i)};\n')
                self.add_reg_str('\t\t\tend if;\n')
        else:
            for i in range(0, array.length):
                self.add_reg_str(f'\t\t\t{array.getNameRead(i)} <= {array.getNameWrite(i)};\n')
        self.add_reg_str('\t\tend if;\n')


    def get_file_suffix(self) -> str:
        return 'vhd'

    def index_var(self, var_name, index):
        return f'{var_name}({index})'

    def slice_var(self, var_name, high, low):
        return f'{var_name}({high} downto {low})'

    @staticmethod
    def int_to_str(din: int, size=None, meta=None) -> str:
        if meta is not None and meta.type == Type.ARITH:
            return str(din)
        
        if size == None:
            if din:
                return "'1'"
            else:
                return "'0'"
        else:
            return f'"{Emitter._int_to_bin(din, size)}"'

    @staticmethod
    def mask_less(din, size) -> str:
        """
        Example:
            MaskLess(3, 5)  # Output: "00111"
            MaskLess(2, 6)  # Output: "000011"
            MaskLess(5, 5)  # Output: "11111"
            MaskLess(0, 4)  # Output: "0000"
        """
        if (din > size):
            raise ValueError("Unknown value!")
        return '\"' + '0'*(size-din) + '1'*din + '\"'
    
    @staticmethod
    def new() -> Emitter:
        return VHDLEmitter()
    
    @staticmethod
    def mux_index(din, sel) -> str:
        """
        Generate a VHDL array-index expression for selecting an element
        """
        return f'{din.getNameRead()}(to_integer(unsigned({sel.getNameRead()})))'
    
    def add_custom_statement(self, custom_statement):
        for line in custom_statement.vhdl_str.splitlines():
            self.statementString += self.get_current_indent() + line + '\n'
