import re

ATTR_SIZE = "size"
ATTR_SLOTS = "num_slots"
ATTR_TRANSPARENT = "transparent"
ATTR_VALUE = "value"
ATTR_LATENCY = "latency"
ATTR_PREDICATE = "predicate"
ATTR_ABSTRACT_DATA = "abstract_data"
ATTR_IS_DOUBLE = "is_double"
ATTR_BITWIDTH = "bitwidth"
ATTR_IN_BITWIDTH = "input_bitwidth"
ATTR_OUT_BITWIDTH = "output_bitwidth"
ATTR_DATA_BITWIDTH = "data_bitwidth"
ATTR_INDEX_BITWIDTH = "index_bitwidth"
ATTR_ADDR_BITWIDTH = "addr_bitwidth"


class SmvScalarType:

    bitwidth: int
    smv_type: str

    def __init__(self, bitwidth: int):
        """
        Constructor for SmvScalarType.
        """
        self.bitwidth = bitwidth
        self.smv_type = f"unsigned word [{bitwidth}]"

    def format_constant(self, value) -> str:
        """
        Formats a given constant value based on the type.
        """
        if self.bitwidth == 1:
            return "TRUE" if bool(value) else "FALSE"
        else:
            return f"0ud{self.bitwidth}_{value}"

    def __str__(self):
        return f"{self.smv_type}"
