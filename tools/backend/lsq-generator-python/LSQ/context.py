# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class VHDLContext:
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
        self.portInitString = ''
        self.regInitString = ''

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
