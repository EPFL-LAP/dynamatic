from typing import TypedDict, Literal, NotRequired

ExtraSignals = dict[str, int]


class BasePort(TypedDict):
  name: str
  bitwidth: int
  extra_signals: NotRequired[ExtraSignals]


class SinglePort(BasePort):
  array: NotRequired[Literal[False]]


class ArrayPort(BasePort):
  array: Literal[True]
  size: int
  extra_signals_list: NotRequired[list[ExtraSignals]]


Port = SinglePort | ArrayPort

Direction = Literal["in", "out"]
