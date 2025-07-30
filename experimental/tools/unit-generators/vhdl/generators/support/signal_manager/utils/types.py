from typing import TypedDict, NotRequired
from generators.support.utils import ExtraSignals


class Channel(TypedDict):
    name: str
    bitwidth: int
    extra_signals: NotRequired[ExtraSignals]
    size: NotRequired[int]
