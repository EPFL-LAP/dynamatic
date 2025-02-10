import re

ATTR_DATA_TYPE = "data_type"
ATTR_TIMING = "timing"
ATTR_SIZE = "size"
ATTR_SLOTS = "slots"
ATTR_VALUE = "value"
ATTR_PORT_TYPES = "port_types"


class SmvScalarType:

  mlir_type: str
  bitwidth: int
  floating_point: bool
  signed: bool
  smv_type: str

  def __init__(self, mlir_type: str):
    """
    Constructor for SmvScalarType.
    Parses an incoming MLIR type string.
    """
    self.mlir_type = mlir_type

    control_pattern = "!handshake.control<>"
    channel_pattern = r"^!handshake\.channel<([u]?i|f)(\d+)>$"
    match = re.match(channel_pattern, mlir_type)

    if mlir_type == control_pattern:
      self.bitwidth = 0
    elif match:
      type_prefix = match.group(1)
      self.bitwidth = int(match.group(2))
      if type_prefix == "f":
        self.floating_point = True
        if self.bitwidth != 32 and self.bitwidth != 64:
          raise ValueError(f"Bitwidth {self.bitwidth} is not supported for floats")
        self.smv_type = "real"
      else:
        self.signed = not type_prefix.startswith("u")
        if self.bitwidth == 1:
          self.smv_type = "boolean"
        elif self.signed:
          self.smv_type = f"signed word [{self.bitwidth}]"
        else:
          self.smv_type = f"unsigned word [{self.bitwidth}]"
    else:
      raise ValueError(f"Type {mlir_type} doesn't correspond to any SMV type")

  def format_constant(self, value) -> str:
    """
    Formats a given constant value based on the type.
    """
    if self.bitwidth == 1:
      return "TRUE" if bool(value) else "FALSE"
    else:
      return int(value)

  def __str__(self):
    return f"{self.smv_type}"


HANDSHAKE_CONTROL_TYPE = SmvScalarType("!handshake.control<>")

def TEHB_BUFFER_PARAMS(data_type):
  return {ATTR_SLOTS: 1, ATTR_TIMING: "R: 1", ATTR_PORT_TYPES: {"outs": data_type.mlir_type}}