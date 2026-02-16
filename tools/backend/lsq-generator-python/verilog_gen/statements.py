from verilog_gen.emitters.emitter import Emitter


class Statement:
    """
    Represents a statement base class.

    This class contains common operator overloads and helpers used across the
    generator. Extracted into its own module to avoid circular imports when
    `signals.py` needs to reference the type.
    """
    def __add__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.ADD, other)

    def __sub__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.SUB, other)

    def __and__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.AND, other)

    def __or__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.OR, other)

    def __xor__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.XOR, other)

    def __mul__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.MUL, other)

    def __invert__(self):
        from verilog_gen.operators.assign import Un, UnOp
        return Un(UnOp.NOT, self)

    def __ge__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.GE, other)

    def __le__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.LE, other)

    def __gt__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.GT, other)

    def __lt__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.LT, other)

    def __eq__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.EQ, other)

    def __ne__(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.NEQ, other)

    def concat(self, other):
        from verilog_gen.operators.assign import Bin, BinOp
        return Bin(self, BinOp.CONCAT, other)

    def to_str(self, em: Emitter, size, super_precedence: int) -> str:
        if self.get_precedence() <= super_precedence:
            return f'({self._to_str(em, size)})'
        else:
            return self._to_str(em, size)

    def when(self, condition):
        self.condition = condition
        return self

    def else_(self, statement):
        if getattr(self, 'condition', None) is None:
            raise ValueError('else_ can only be called after when')

        from verilog_gen.operators.assign import WhenElse
        return WhenElse(self, self.condition, statement)

    def _to_str(self, em: Emitter, size) -> str:
        raise NotImplementedError('Subclasses must implement _to_str method')

    def get_type(self) -> str:
        return 'logic'

    def get_precedence(self) -> int:
        # TODO: Cleaner way to handle precedence
        return 10000
