from verilog_gen.emitters import Emitter
from verilog_gen.ir import Statement, Bin, Un, BinOp, UnOp, Bit
from verilog_gen.signals import Logic, LogicVec, LogicArray, LogicVecArray
# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class VerilogEmitter(Emitter):
    """
    A context object to replace global variables for VHDL code generation.
    Holds indentation level, temporary name counter, and initialization strings.
    """

    def __init__(self):
        # Indentation level for generated code
        self.tabLevel = 1

        # Counter for generating unique temporary names
        self.tempCount = 0

        # Accumulated initialization code sections
        self.signalInitString = ''

        self.PORT_INIT_STR = '(\n\t\tinput rst,\n\t\tinput clk'
        self.PORT_END_STR = '\n\t);'
        self.portInitString = ''

        self.REG_INIT_STR = '\talways @(posedge clk) begin\n'
        self.REG_END_STR = '\tend\n'
        self.regInitString = ''
        self.statementString = ''
        
        self.inst_started = False

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
        self.regInitString += self.get_current_indent() + code + '\n'

    def add_statement(self, code: str):
        self.statementString += self.get_current_indent() + code

    def add_comment(self, comment: str):
        self.statementString += self.get_current_indent() + f'// {comment}\n'

    def add_assignment(self, out, statement: Statement):
        out_str, size = self.assigned_var_to_str(out)
        statement_str = statement.to_str(self, size, -1)
        # Assume we only write to logic types
        self.statementString += self.get_current_indent() + f'assign {out_str} = {statement_str};\n'

    def get_definition_str(self, module_name: str, write_regs=True) -> str:
        return f'module {module_name} ' + \
                self.PORT_INIT_STR + self.portInitString + self.PORT_END_STR + '\n' + \
                '// SIGNAL INIT\n' + self.signalInitString + '\n' + \
                '// STATEMENTS\n' + self.statementString + '\n' + \
                ((self.REG_INIT_STR + self.regInitString + self.REG_END_STR) if write_regs and self.regInitString != '' else '') \
                + 'endmodule\n'

    def start_instantiation(self, module_name:str, instance_name: str = None) -> str:
        if self.inst_started: # Sanity check to prevent overlapping instantiations
            raise ValueError('start_instantiation called while another instantiation is in progress')

        if instance_name is None: instance_name = module_name

        self.inst_started = True
        self.inst_str = f'{self.get_current_indent()}{module_name} {instance_name} (\n'
        self.increase_indent()

    def add_map(self, port_name: str, signal_name: str) -> str:
        if not self.inst_started:
            raise ValueError('add_map can only be called after start_instantiation')
        
        assert isinstance(port_name, str) and isinstance(signal_name, str), "port name and signal name must be strings"

        self.inst_str += f'{self.get_current_indent()}.{port_name}({signal_name}),\n'

    def complete_instantiation(self) -> str:
        self.inst_started = False
        self.decrease_indent()
        self.inst_str += self.get_current_indent() + ');\n'
        self.statementString += self.inst_str
        self.inst_str = ''

    BINOP_STRINGS = {
        BinOp.ADD: '+',
        BinOp.SUB: '-',
        BinOp.AND: '&',
        BinOp.OR: '|',
        BinOp.XOR: '^',
        BinOp.MUL: '*',
        BinOp.GE: '>=',
        BinOp.LE: '<=',
        BinOp.GT: '>',
        BinOp.LT: '<',
        BinOp.EQ: '==',
        BinOp.NEQ: '!=',
    }

    def get_binop_str(self, op: Bin) -> str:
        if op in self.BINOP_STRINGS:
            return self.BINOP_STRINGS[op]
        else:
            raise ValueError('Invalid binary operator: ' + str(op))
            
    def get_unop_str(self, unop: UnOp) -> str:
        if unop == UnOp.NOT:
            return '!'
        else:
            raise ValueError('Invalid unary operator')

    def get_bit_str(self, bit: Bit) -> str:
        if bit.value == 0:
            return '1\'b0'
        elif bit.value == 1:
            return '1\'b1'
        else:
            raise ValueError('Invalid bit value')

    def bin_to_str(self, bin: Bin, size: int) -> str:
        left_str = bin.left.to_str(self, size, bin.get_precedence())
        right_str = bin.right.to_str(self, size, bin.get_precedence())

        if bin.op == BinOp.CONCAT:
            return f'{{{left_str}, {right_str}}}'

        return f'{left_str} {self.get_binop_str(bin.op)} {right_str}'

    def un_to_str(self, un: Un, size: int) -> str:
        val_str = un.val.to_str(self, size, un.get_precedence())
        return f'{self.get_unop_str(un.op)} {val_str}'

    def when_else_to_str(self, when_else, size: int) -> str:
        true_str = when_else.true_statement.to_str(self, size, 0)
        false_str = when_else.false_statement.to_str(self, size, 0)
        cond_str = when_else.condition.to_str(self, size, 0)

        return f'{cond_str} ? {true_str} : {false_str}'


    def logic_signal_init(self, signal: Logic, sufix: str):
        """
        Appends the appropriate declaration or port line for this signal to a global buffer.
        """
        if (signal.type == 'w'):
            self.add_signal_str(f'\twire {signal.name + sufix};\n')
        elif (signal.type == 'r'):
            self.add_signal_str(f'\treg {signal.name + sufix}_d;\n')
            self.add_signal_str(f'\treg {signal.name + sufix}_q;\n')
        elif (signal.type == 'i'):
            self.add_port_str(',\n')
            self.add_port_str(f'\t\tinput {signal.name + sufix}_i')
        elif (signal.type == 'o'):
            self.add_port_str(',\n')
            self.add_port_str(f'\t\toutput {signal.name + sufix}_o')


    def logicvec_signal_init(self, vec: LogicVec, sufix: str):
        if (vec.type == 'w'):
            self.add_signal_str(f'\twire [{vec.size-1}:0] {vec.name + sufix};\n')
        elif (vec.type == 'r'):
            self.add_signal_str(f'\treg [{vec.size-1}:0] {vec.name + sufix}_d;\n')
            self.add_signal_str(f'\treg [{vec.size-1}:0] {vec.name + sufix}_q;\n')
        elif (vec.type == 'i'):
            self.add_port_str(',\n')
            self.add_port_str(f'\t\tinput [{vec.size-1}:0] {vec.name + sufix}_i')
        elif (vec.type == 'o'):
            self.add_port_str(',\n')
            self.add_port_str(f'\t\toutput [{vec.size-1}:0] {vec.name + sufix}_o')
    

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
        self.increase_indent()
        in_else = False
        if (init != None):
            self.add_reg_str('if (rst)')
            self.add_reg_str(f'\t{logic.getNameRead()} <= {self.int_to_bits(init)};')
            self.add_reg_str('else')
            in_else = True
            self.increase_indent()

        if (enable != None):
            self.add_reg_str(f'if ({enable.getNameRead()})')
            self.add_reg_str(f'\t{logic.getNameRead()} <= {logic.getNameWrite()};')
            self.add_reg_str('end')
        else:
            self.add_reg_str(f'{logic.getNameRead()} <= {logic.getNameWrite()};')

        if in_else:
            self.decrease_indent()
            self.add_reg_str('end')
        self.decrease_indent()

    def logicvec_reg_init(self, vec: LogicVec, enable=None, init=None) -> None:
        assert (vec.type == 'r')
        self.increase_indent()
        in_else = False
        if (init != None):
            self.add_reg_str('if (rst)')
            self.add_reg_str(f'\t{vec.getNameRead()} <= {self.int_to_bits(init, vec.size)};')
            self.add_reg_str('else')
            in_else = True
            self.increase_indent()

        if (enable != None):
            self.add_reg_str(f'if ({enable.getNameRead()})')
            self.add_reg_str(f'\t{vec.getNameRead()} <= {vec.getNameWrite()};')
            self.add_reg_str('end')
        else:
            self.add_reg_str(f'{vec.getNameRead()} <= {vec.getNameWrite()};')

        if in_else:
            self.decrease_indent()
            self.add_reg_str('end')
        self.decrease_indent()



    def logicarray_reg_init(self, array: LogicArray, enable=None, init=None) -> None:
        assert (array.type == 'r')
        self.increase_indent()
        in_else = False
        if (init != None):
            self.add_reg_str('if (rst)')
            for i in range(0, array.length):
                self.add_reg_str(f'\t{array.getNameRead(i)} <= {self.int_to_bits(init[i])};')
            self.add_reg_str('else')
            in_else = True
            self.increase_indent()

        if (enable != None):
            for i in range(0, array.length):
                self.add_reg_str(f'if ({enable.getNameRead(i)})')
                self.add_reg_str(f'\t{array.getNameRead(i)} <= {array.getNameWrite(i)};')
                self.add_reg_str('end')
        else:
            for i in range(0, array.length):
                self.add_reg_str(f'{array.getNameRead(i)} <= {array.getNameWrite(i)};')

        if in_else:
            self.decrease_indent()
            self.add_reg_str('end')
        self.decrease_indent()


    def logicvecarray_reg_init(self, array: LogicVecArray, enable=None, init=None) -> None:
        assert (array.type == 'r')
        self.increase_indent()
        in_else = False
        if (init != None):
            self.add_reg_str('if (rst)')
            for i in range(0, array.length):
                self.add_reg_str(f'\t{array.getNameRead(i)} <= {self.int_to_bits(init[i], array.size)};')
            self.add_reg_str('else')
            in_else = True
            self.increase_indent()

        if (enable != None):
            for i in range(0, array.length):
                self.add_reg_str(f'if ({enable.getNameRead(i)})')
                self.add_reg_str(f'\t{array.getNameRead(i)} <= {array.getNameWrite(i)};')
                self.add_reg_str('end')
        else:
            for i in range(0, array.length):
                self.add_reg_str(f'{array.getNameRead(i)} <= {array.getNameWrite(i)};')

        if in_else:
            self.decrease_indent()
            self.add_reg_str('end')
        self.decrease_indent()


    def get_file_suffix(self) -> str:
        return 'v'

    def index_var(self, var_name, index):
        return f'{var_name}[{index}]'

    def slice_var(self, var_name, high, low):
        return f'{var_name}[{high}:{low}]'

    @staticmethod
    def int_to_bits(din, size=None) -> str:
        if size == None:
            size = 1

        return f'{size}\'b{Emitter._int_to_bin(din, size)}'
        
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
        return f'{size}\'b' + '0'*(size-din) + '1'*din
    
    @staticmethod
    def new() -> Emitter:
        return VerilogEmitter()
    
    def mux_index(self, din, sel) -> str:
        """
        Generate a Verilog array-index expression for selecting an element
        """
        return f'{din.getNameRead()}[{sel.getNameRead()}]'