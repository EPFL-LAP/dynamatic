class ExtraSignalMapping:
  # List of tuples of (extra_signal_name, (msb, lsb))
  mapping: list[tuple[str, tuple[int, int]]]
  total_bitwidth: int

  def __init__(self, offset: int = 0):
    """
    offset: The starting bitwidth of the extra signals (if data is present).
    """
    self.mapping = []
    self.total_bitwidth = offset

  def add(self, name: str, bitwidth: int):
    self.mapping.append(
        (name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
    self.total_bitwidth += bitwidth

  def has(self, name: str) -> bool:
    return name in [name for name, _ in self.mapping]

  def get(self, name: str):
    return self.mapping[[name for name, _ in self.mapping].index(name)]


def get_default_extra_signal_value(extra_signal_name: str):
  return "\"0\""


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
  return sum(extra_signals.values())
