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

