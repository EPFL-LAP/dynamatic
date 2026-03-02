from enum import Enum


class Statement:
    """
    Represents a statement base class.

    This class contains common operator overloads and helpers used across the
    generator.
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

    def to_str(self, em: "Emitter", meta: "Meta") -> str:
        if self.get_precedence() <= meta.precedence:
            return f"({self._to_str(em, meta)})"
        else:
            return self._to_str(em, meta)

    def when(self, condition):
        self.condition = condition
        return self

    def else_(self, statement):
        if getattr(self, "condition", None) is None:
            raise ValueError("else_ can only be called after when")

        return WhenElse(self, self.condition, statement)

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        raise NotImplementedError("Subclasses must implement _to_str method")

    def get_type(self) -> str:
        return Type.LOGIC

    def get_precedence(self) -> int:
        # TODO: Cleaner way to handle precedence
        return 10000


class Type(Enum):
    LOGIC = "logic"
    ARITH = "arith"
    BOOL = "bool"
    ANY = "any"


class Val(Statement):
    """
    Represents a value, which can be either a variable or a constant
    Can be initialized with a string, an int, a tuple of (var, index), or a tuple of (var, index1, index2)
    """

    def __init__(self, *var, size=None):
        self.var = var
        self.size = size

    def _to_str(self, em: "Emitter", meta: "Meta"):
        if self.size is not None:
            size = self.size
        else:
            size = meta.size

        arg = self.var[0] if len(self.var) == 1 else tuple(self.var)
        if type(arg) == str:
            str_ret = arg
        elif type(arg) == int:
            str_ret = em.int_to_str(arg, size, meta=meta)
        elif type(arg) == tuple:
            if type(arg[0]) == int:
                str_ret = em.int_to_str(arg[0], arg[1], meta=meta)
            elif len(arg) == 2:
                str_ret = arg[0].getNameRead(arg[1])
            else:
                str_ret = arg[0].getNameRead(arg[1], arg[2])
        else:
            str_ret = arg.getNameRead()

        return str_ret

    def get_type(self) -> str:
        return Type.ANY if type(self.var[0]) == int else Type.LOGIC


class BinOp(Enum):
    """Represents a binary operator"""

    ADD = ("+", 4, Type.ARITH, Type.ARITH)
    SUB = ("-", 4, Type.ARITH, Type.ARITH)
    AND = ("and", 3, Type.LOGIC, Type.LOGIC)
    OR = ("or", 3, Type.LOGIC, Type.LOGIC)
    XOR = ("xor", 3, Type.LOGIC, Type.LOGIC)
    CONCAT = ("&", 3, Type.LOGIC, Type.LOGIC)
    MUL = ("*", 5, Type.ARITH, Type.ARITH)
    GE = (">=", 2, Type.BOOL, Type.ARITH)
    LE = ("<=", 2, Type.BOOL, Type.ARITH)
    GT = (">", 2, Type.BOOL, Type.ARITH)
    LT = ("<", 2, Type.BOOL, Type.ARITH)
    EQ = ("=", 1, Type.BOOL, Type.ANY)
    NEQ = ("!=", 1, Type.BOOL, Type.ANY)

    def get_precedence(self) -> int:
        return self.value[1]

    def get_type(self) -> str:
        return self.value[2]

    def get_param_type(self) -> str:
        return self.value[3]


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

    def get_param_type(self) -> str:
        return self.op.get_param_type()

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        return em.bin_to_str(self, meta)


class UnOp(Enum):
    """Represents a unary operator"""

    NOT = ("not", 10, Type.LOGIC, Type.LOGIC)

    def get_precedence(self) -> int:
        return self.value[1]

    def get_type(self) -> str:
        return self.value[2]

    def get_param_type(self) -> str:
        return self.value[3]


class Un(Statement):
    """
    Represents a unary statement
    """

    def __init__(self, op: UnOp, val: Statement):
        self.op = op
        self.val = val

    def get_precedence(self) -> int:
        return self.op.get_precedence()

    def get_type(self) -> str:
        return self.op.get_type()

    def get_param_type(self) -> str:
        return self.op.get_param_type()

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        return em.un_to_str(self, meta)


class Bit(Statement):
    """
    Represents a bit statement
    """

    def __init__(self, value: int):
        if value not in (0, 1):
            raise ValueError("Bit value must be 0 or 1")
        self.value = value

    def get_precedence(self) -> int:
        return 11

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        return em.get_bit_str(self)


class CustomStatement(Statement):
    """
    Represents a custom string statement
    """

    def __init__(self, vhdl_str, verilog_str):
        self.vhdl_str = vhdl_str
        self.verilog_str = verilog_str

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        return em.print_custom_str(self)


class WhenElse(Statement):
    """
    Represents a when-else statement
    """

    def __init__(
        self,
        true_statement: Statement,
        condition: Statement,
        false_statement: Statement,
    ):
        self.condition = condition
        self.true_statement = true_statement
        self.false_statement = false_statement

    def get_precedence(self) -> int:
        return 0

    def _to_str(self, em: "Emitter", meta: "Meta") -> str:
        return em.when_else_to_str(self, meta)

    def get_type(self) -> str:
        if (
            self.true_statement.get_type() != self.false_statement.get_type()
            and self.true_statement.get_type() != Type.ANY
            and self.false_statement.get_type() != Type.ANY
        ):
            raise ValueError(
                f"true_statement and false_statement must have the same type, got {self.true_statement.get_type()} and {self.false_statement.get_type()}"
            )
        return self.true_statement.get_type()
