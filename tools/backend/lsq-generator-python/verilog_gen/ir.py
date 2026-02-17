from enum import Enum

class Statement:
    """
    Represents a statement base class.

    This class contains common operator overloads and helpers used across the
    generator. Extracted into its own module to avoid circular imports when
    `signals.py` needs to reference the type.
    """
    def __add__(self, other):
        return Bin(self, BinOp.ADD, other)

    def __sub__(self, other):
        return Bin(self, BinOp.SUB, other)

    def __and__(self, other):
        return Bin(self, BinOp.AND, other)

    def __or__(self, other):
        return Bin(self, BinOp.OR, other)

    def __xor__(self, other):
        return Bin(self, BinOp.XOR, other)

    def __mul__(self, other):
        return Bin(self, BinOp.MUL, other)

    def __invert__(self):
        return Un(UnOp.NOT, self)

    def __ge__(self, other):
        return Bin(self, BinOp.GE, other)

    def __le__(self, other):
        return Bin(self, BinOp.LE, other)

    def __gt__(self, other):
        return Bin(self, BinOp.GT, other)

    def __lt__(self, other):
        return Bin(self, BinOp.LT, other)

    def __eq__(self, other):
        return Bin(self, BinOp.EQ, other)

    def __ne__(self, other):
        return Bin(self, BinOp.NEQ, other)

    def concat(self, other):
        return Bin(self, BinOp.CONCAT, other)

    def to_str(self, em: 'Emitter', size, super_precedence: int) -> str:
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

        return WhenElse(self, self.condition, statement)

    def _to_str(self, em: 'Emitter', size) -> str:
        raise NotImplementedError('Subclasses must implement _to_str method')

    def get_type(self) -> str:
        return 'logic'

    def get_precedence(self) -> int:
        # TODO: Cleaner way to handle precedence
        return 10000

class Val(Statement):
    """
    Represents a variable statement
    """
    def __init__(self, *var):
        self.var = var

    def _to_str(self, em: 'Emitter', size):
        arg = self.var[0] if len(self.var) == 1 else tuple(self.var)
        if type(arg) == str:
            str_ret = arg
        elif type(arg) == int:
            str_ret = em.int_to_bits(arg, size)
        elif type(arg) == tuple:
            if type(arg[0]) == int:
                str_ret = em.int_to_bits(arg[0], arg[1])
            elif len(arg) == 2:
                str_ret = arg[0].getNameRead(arg[1])
            else:
                str_ret = arg[0].getNameRead(arg[1], arg[2])
        else:
            str_ret = arg.getNameRead()

        return str_ret


class BinOp(Enum):
    ADD = ('+', 4, 'arith')
    SUB = ('-', 4, 'arith')
    AND = ('and', 3, 'logic')
    OR = ('or', 3, 'logic')
    XOR = ('xor', 3, 'logic')
    CONCAT = ('&', 3, 'logic')
    MUL = ('*', 5, 'arith')
    GE = ('>=', 2, 'bool')
    LE = ('<=', 2, 'bool')
    GT = ('>', 2, 'bool')
    LT = ('<', 2, 'bool')
    EQ = ('=', 1, 'bool')
    NEQ = ('!=', 1, 'bool')

    def get_precedence(self) -> int:
        return self.value[1]

    def get_type(self) -> str:
        return self.value[2]

class Bin(Statement):
    """
    Represents a binary statement
    """
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
    def get_precedence(self) -> int:
        return self.op.get_precedence()

    def get_type(self) -> str:
        return self.op.get_type()

    def _to_str(self, em: 'Emitter', size: int) -> str:
        return em.bin_to_str(self, size)

class UnOp(Enum):
    NOT = ('not', 10, 'logic')

    def get_precedence(self) -> int:
        if self == UnOp.NOT:
            return 10
        else:
            raise ValueError('Invalid unary operator')

class Un(Statement):
    """
    Represents a unary statement
    """
    def __init__(self, op: UnOp, val: Statement):
        self.op = op
        self.val = val


    def get_precedence(self) -> int:
        return self.op.get_precedence()

    def _to_str(self, em: 'Emitter', size: int) -> str:
        return em.un_to_str(self, size)

class Bit(Statement):
    """
    Represents a bit statement
    """
    def __init__(self, value: int):
        if value not in (0, 1):
            raise ValueError('Bit value must be 0 or 1')
        self.value = value

    def get_precedence(self) -> int:
        return 11

    def _to_str(self, em: 'Emitter', size) -> str:
        return em.get_bit_str(self)

class CustomStr(Statement):
    """
    Represents a custom string statement
    """
    def __init__(self, vhdl_str, verilog_str):
        self.vhdl_str = vhdl_str
        self.verilog_str = verilog_str

    def _to_str(self, em: 'Emitter', size) -> str:
        return em.print_custom_str(self)

class WhenElse(Statement):
    """
    Represents a when-else statement
    """
    def __init__(self, true_statement: Statement, condition: Statement,  false_statement: Statement):
        self.condition = condition
        self.true_statement = true_statement
        self.false_statement = false_statement

    def get_precedence(self) -> int:
        return 0

    def _to_str(self, em: 'Emitter', size) -> str:
        return em.when_else_to_str(self, size)

    def get_type(self) -> str:
        if self.true_statement.get_type() != self.false_statement.get_type():
            raise ValueError('true_statement and false_statement must have the same type')
        return self.true_statement.get_type()
        
def Op(ctx, *args):
    raise ValueError('Op is deprecated, please use Val, Bin, Un, WhenElse, or CustomStr instead')