import re




class SmvScalarType:

  mlir_type: str
  bitwidth: int
  signed: bool
  smv_type: str

  def __init__(self, mlir_type: str):
    """
    Constructor for SmvScalarType.
    Parses an incoming MLIR type string.
    """
    self.mlir_type = mlir_type

    control_pattern = "!handshake.control<>"
    channel_pattern = r"^!handshake\.channel<([u]?i)(\d+)>$"
    match = re.match(channel_pattern, mlir_type)

    if mlir_type == control_pattern:
      self.bitwidth = 0
    elif match:
      self.signed = not match.group(1).startswith("u")
      self.bitwidth = int(match.group(2))
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


HANSHAKE_CONTROL_TYPE = SmvScalarType("!handshake.control<>")