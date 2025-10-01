from typing import Type, Union
from enum import Enum

ExtraSignals = dict[str, int]


def try_enum_cast(value: str, enum_class: Type[Enum]) -> Union[Enum, str]:
    try:
        return enum_class(value)
    except ValueError:
        return value