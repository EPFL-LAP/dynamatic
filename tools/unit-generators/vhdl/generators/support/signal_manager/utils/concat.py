from .types import ExtraSignals

# Holds the concatenation layout of extra signals.
# Tracks each signal's bit range and keeps the total combined bitwidth.


class ConcatLayout:
    # List of tuples of (extra_signal_name, (msb, lsb))
    # e.g., [("spec", (0, 0)), ("tag0", (8, 1))]
    mapping: list[tuple[str, tuple[int, int]]]
    total_bitwidth: int

    def __init__(self, extra_signals: dict[str, int]):
        self.mapping = []
        self.total_bitwidth = 0

        for name, bitwidth in extra_signals.items():
            self.add(name, bitwidth)

    def add(self, name: str, bitwidth: int):
        self.mapping.append(
            (name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
        self.total_bitwidth += bitwidth

    @property
    def extra_signals(self) -> ExtraSignals:
        return {name: msb - lsb + 1 for name, (msb, lsb) in self.mapping}


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
    return sum(extra_signals.values())
