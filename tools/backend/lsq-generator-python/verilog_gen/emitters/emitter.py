# ===----------------------------------------------------------------------===#
# Global Parameter Initialization
# ===----------------------------------------------------------------------===#
class Emitter:
    """
    A context object to replace global variables for code generation.
    Holds indentation level, temporary name counter, and initialization strings.
    """

    def __init__(self):
        raise NotImplementedError("Emitter is an abstract class and cannot be instantiated directly.")

    def get_current_indent(self) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_current_indent()")

    def increase_indent(self):
        raise NotImplementedError("Emitter subclasses must implement increase_indent()")


    def decrease_indent(self):
        raise NotImplementedError("Emitter subclasses must implement decrease_indent()")

    def get_temp(self, name: str) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_temp()")

    def use_temp(self):
        raise NotImplementedError("Emitter subclasses must implement use_temp()")

    def add_signal_str(self, code: str):
        raise NotImplementedError("Emitter subclasses must implement add_signal_str()")

    def add_port_str(self, code: str):
        raise NotImplementedError("Emitter subclasses must implement add_port_str()")

    def add_reg_str(self, code: str):
        raise NotImplementedError("Emitter subclasses must implement add_reg_str()")

    def add_statement(self, code: str):
        raise NotImplementedError("Emitter subclasses must implement add_statement()")

    def add_assignment(self, op: str):
        raise NotImplementedError("Emitter subclasses must implement add_assignment()")

    def comment(self, op: str):
        raise NotImplementedError("Emitter subclasses must implement add_comment()")

    def get_binop_str(self, op) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_binop_str()")

    def get_unop_str(self, op) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_unop_str()")

    def get_bit_str(self, bit, size: int) -> str:
        raise NotImplementedError("Emitter subclasses must implement get_bit_str()")

    def bin_to_str(self, bin, size: int) -> str:
        raise NotImplementedError("Emitter subclasses must implement bin_to_str()")

    def un_to_str(self, un, size: int) -> str:
        raise NotImplementedError("Emitter subclasses must implement un_to_str()")

    def assigned_var_to_str(self, var):
        from verilog_gen.signals import Logic
        size = 1
        if type(var) == tuple:
            if len(var) == 2:
                str_ret = f'{var[0].getNameWrite(var[1])}'
            else:
                str_ret = f'{var[0].getNameWrite(var[1], var[2])}'
        else:
            str_ret = f'{var.getNameWrite()}'
            if (type(var) != Logic):
                size = var.size

        return str_ret, size
