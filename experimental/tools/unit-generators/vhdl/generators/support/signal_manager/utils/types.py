from typing import TypedDict, NotRequired

# Define the type for extra signals, which are stored as a dictionary with signal names and their bitwidths.
ExtraSignals = dict[str, int]


class Channel(TypedDict):
    name: str
    bitwidth: int
    extra_signals: NotRequired[ExtraSignals]
    size: NotRequired[int]
