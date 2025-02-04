import re

# todo: change from class?
class VhdlScalarType:

  mlir_type: str
  # Note: VHDL only require information on bitwidth and extra signals
  bitwidth: int
  extra_signals: dict[str, int] # key: name, value: bitwidth (todo)

  def __init__(self, mlir_type: str):
    """
    Constructor for VhdlScalarType.
    Parses an incoming MLIR type string.
    """
    self.mlir_type = mlir_type

    control_pattern = "!handshake.control<>"
    channel_pattern = r"^!handshake\.channel<([u]?i)(\d+)>$"
    match = re.match(channel_pattern, mlir_type)

    if mlir_type == control_pattern:
      self.bitwidth = 0
    elif match:
      self.bitwidth = int(match.group(2))
    else:
      raise ValueError(f"Type {mlir_type} is invalid")

  def has_extra_signals(self):
    pass # todo

  def is_channel(self):
    pass # todo
