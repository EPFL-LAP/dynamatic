# pyright: reportInvalidTypeForm=false

from amaranth import *
from amaranth.lib.data import ArrayLayout
from amaranth.lib.wiring import Component, In, Out

from functools import reduce


class MuxOneHot(Component):
    """One-hot multiplexer.

    output_o is computed as the OR-reduction of the input elements, each masked
    (AND'd) with its corresponding bit of sel_i:

        output_o = OR_i( input_i[i] & Repl(sel_oh_i[i], width) )

    If sel_oh_i is strictly one-hot, this selects exactly one element. If
    multiple bits of sel_oh_i are set, the selected elements are OR'd together.
    If no bits are set, out_o is 0.

    Parameters
    ----------
    shape : shape-like
        Shape of each input element, and of output_o.
    depth : int
        Number of input elements (== width of sel_oh_i).
    """

    def __init__(self, shape, depth):
        self._shape = shape
        self._depth = depth
        super().__init__({
            "input_i":  In(ArrayLayout(shape, depth)),
            "sel_oh_i": In(depth),
            "output_o": Out(shape),
        })

    def elaborate(self, platform):
        m = Module()

        masked = [
            Mux(self.sel_oh_i[i], self.input_i[i], 0)
            for i in range(self._depth)
        ]
        m.d.comb += self.output_o.eq(reduce(lambda a, b: a | b, masked))

        return m


class RotateLeft(Component):
    """Rotate a vector left by a specified (variable) amount.

    Parameters
    ----------
    shape: shape-like
        Shape of the input and output vectors.
    depth: int
        Depth of the input and output vectors.
    rotate_amount_width: int
        Width of the binary vector representing the rotate amount.
    """

    def __init__(self, shape, depth, rotate_amount_width):
        self._shape = shape
        self._depth = depth
        self._rotate_amount_width = rotate_amount_width
        super().__init__({
            "input_i": In(ArrayLayout(shape, depth)),
            "rotate_amount_i": In(rotate_amount_width),
            "output_o": Out(ArrayLayout(shape, depth)),
        })

    def elaborate(self, platform):
        m = Module()

        input_double = Signal(ArrayLayout(self._shape, self._depth * 2))
        m.d.comb += input_double.eq(Cat(self.input_i, self.input_i))

        # Select the appropriate rotated version based on the rotate_amount_i input.
        with m.Switch(self.rotate_amount_i):
            for i in range(self._depth):
                begin = self._depth - i
                assert begin >= 0, "Rotation begin index must be non-negative"
                rotated = input_double[begin:begin + self._depth]
                with m.Case(i):
                    m.d.comb += self.output_o.eq(rotated)
            with m.Default():
                m.d.comb += self.output_o.eq(0)

        return m


class WrapSubtract(Component):
    def __init__(self, width: int, limit: int):
        assert 0 < limit <= 2 ** width, "limit must be in the range (0, 2^width]"
        self.width = width
        self.limit = limit
        super().__init__({
            "in_i": In(unsigned(width)),
            "sub_i": In(unsigned(width)),
            "out_o": Out(unsigned(width)),
        })

    def elaborate(self, platform):
        m = Module()

        if self.limit == 2 ** self.width:
            # Regular subtraction will wrap around automatically
            m.d.comb += self.out_o.eq(self.in_i - self.sub_i)
        else:
            # Perform subtraction with wraparound behavior.
            with m.If(self.in_i >= self.sub_i):
                m.d.comb += self.out_o.eq(self.in_i - self.sub_i)
            with m.Else():
                limit_const = Const(self.limit, unsigned(self.width))
                m.d.comb += self.out_o.eq(self.in_i + limit_const - self.sub_i)

        return m
