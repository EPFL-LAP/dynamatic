from typing import TypedDict, Literal, NotRequired

# Define the type for extra signals, which are stored as a dictionary with signal names and their bitwidths.
ExtraSignals = dict[str, int]


class Port(TypedDict):
  name: str
  bitwidth: int
  extra_signals: NotRequired[ExtraSignals]
  size: NotRequired[int]


# Define the direction for ports (either 'in' or 'out').
Direction = Literal["in", "out"]
