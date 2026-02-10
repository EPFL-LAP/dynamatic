from verilog_gen.emitters.emitter import Emitter
# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class VHDLEmitter(Emitter):
    """
    A context object to replace global variables for VHDL code generation.
    Holds indentation level, temporary name counter, and initialization strings.
    """

    def __init__(self):
        # Indentation level for generated code
        self.tabLevel = 0

        # Counter for generating unique temporary names
        self.tempCount = 0

        # Accumulated initialization code sections
        self.signalInitString = ''

        self.PORT_INIT_STR = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        self.PORT_END_STR = '\n\t);'
        self.portInitString = ''

        self.REG_INIT_STR = '\tprocess (clk, rst) is\n\tbegin\n'
        self.REG_END_STR = '\tend process;\n'
        self.regInitString = ''
        self.statementString = ''

        # Default library imports for VHDL
        self.library = 'library IEEE;\nuse IEEE.std_logic_1164.all;\nuse IEEE.numeric_std.all;\n\n'

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

    def add_assignment(self, op: str):
        self.statementString += op;

    def to_string(self, module_name: str, write_regs=True) -> str:
        return self.library + \
                f'entity {module_name} is\n' + \
                self.PORT_INIT_STR + self.portInitString + self.PORT_END_STR + \
                '\nend entity;\n\n' + \
                f'architecture arch of {module_name} is\n' + \
                self.signalInitString + \
                'begin\n' + self.statementString + '\n' + \
                ((self.REG_INIT_STR + self.regInitString + self.REG_END_STR) if write_regs else '') \
                + 'end architecture;\n'