from typing import TypedDict, Literal, NotRequired

# Define the type for extra signals, which are stored as a dictionary with signal names and their bitwidths.
ExtraSignals = dict[str, int]


# Base class for port, shared by both SinglePort and ArrayPort.
class BasePort(TypedDict):
  name: str  # Port name (e.g., 'data_in')
  bitwidth: int  # Port bitwidth (e.g., 8, 32)
  extra_signals: NotRequired[ExtraSignals]  # Optional extra signals (if any)


# SinglePort represents a port that is not an array.
class SinglePort(BasePort):
  array: NotRequired[Literal[False]]  # Indicates this is not an array port


# ArrayPort represents a port that is an array.
class ArrayPort(BasePort):
  array: Literal[True]  # Confirms this is an array port
  size: int  # Size of the array (e.g., 4, 8)
  # Optional list of extra signals for each array element
  extra_signals_list: NotRequired[list[ExtraSignals]]


# Port can either be a SinglePort or ArrayPort.
Port = SinglePort | ArrayPort

# Define the direction for ports (either 'in' or 'out').
Direction = Literal["in", "out"]
