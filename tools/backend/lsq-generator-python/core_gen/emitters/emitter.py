# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class Emitter:
    """
    A context object to replace global variables for code generation.
    Holds indentation level, temporary name counter, and initialization strings.
    """

    def __init__(self, clock_name: str = "clk", reset_name: str = "rst"):
        self.tabLevel = 1
        self.tempCount = 0

        self.signalInitString = ""
        self.portInitString = ""
        self.regInitString = ""
        self.statementString = ""

        self.clock_name = clock_name
        self.reset_name = reset_name

        self.inst_started = False

        # Keep Emitter abstract: prevent direct instantiation of the base class
        if self.__class__ is Emitter:
            raise NotImplementedError(
                "Emitter is an abstract class and cannot be instantiated directly."
            )

    def get_current_indent(self) -> str:
        return "\t" * self.tabLevel

    def increase_indent(self):
        self.tabLevel += 1

    def decrease_indent(self):
        self.tabLevel = max(0, self.tabLevel - 1)

    def get_temp(self, name: str) -> str:
        return f"TEMP_{self.tempCount}_{name}"

    def use_temp(self):
        self.tempCount += 1

    def add_signal_str(self, code: str):
        self.signalInitString += code

    def add_port_str(self, code: str):
        self.portInitString += code

    def add_statement(self, code: str):
        self.statementString += self.get_current_indent() + code

    def add_reg_str(self, code: str):
        raise NotImplementedError("Emitter subclasses must implement add_reg_str()")

    def add_assignment(self, out, statement: 'Statement', in_process: bool = False):
        raise NotImplementedError("Emitter subclasses must implement add_assignment()")

    def add_comment(self, comment: str):
        raise NotImplementedError("Emitter subclasses must implement add_comment()")

    def get_binop_str(self, op) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_binop_str()")

    def get_unop_str(self, op) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_unop_str()")

    def get_bit_str(self, bit) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_bit_str()")

    def bin_to_str(self, bin, meta: 'Meta') -> str:
        raise NotImplementedError("Emitter subclasses must implement bin_to_str()")

    def un_to_str(self, un, meta: 'Meta') -> str:
        raise NotImplementedError("Emitter subclasses must implement un_to_str()")

    def assigned_var_to_str(self, var):
        from core_gen.signals import Logic

        size = 1
        if type(var) == tuple:
            if len(var) == 2:
                str_ret = f"{var[0].getNameWrite(var[1])}"
            else:
                str_ret = f"{var[0].getNameWrite(var[1], var[2])}"
        else:
            str_ret = f"{var.getNameWrite()}"
            if type(var) != Logic:
                size = var.size

        return str_ret, size

    @staticmethod
    def _int_to_bin(val: int, size: int) -> str:
        """
        Converts an integer to a binary string of the specified size.
        Example:
            int_to_bin(5, 8)  # Output: 00000101
            int_to_bin(10, 4) # Output: 1010
            int_to_bin(3, 3)  # Output: 011
            int_to_bin(0, 5)  # Output: 00000
        """
        if val < 0 or val >= (1 << size):
            raise ValueError(f"Value {val} out of range for the specified size {size}")

        return f"{val:0{size}b}"


class Meta:
    """
    Contains the meta necessary to generate correct sub statements
    """

    def __init__(self, size, statement_type, precedence):
        self.size = size
        self.type = statement_type
        self.precedence = precedence
