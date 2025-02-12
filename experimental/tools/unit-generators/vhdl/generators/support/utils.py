import re

def parse_extra_signals(extra_signals: str) -> dict[str, int]:
  """
  Parses a string of extra signals and their bitwidths.
  e.g., extra_signals = "spec: i1, tag: i8"
  """
  type_pattern = r"u?i(\d+)"
  extra_signals_dict = {}
  for signal in extra_signals.split(","):
    name, signal_type = signal.split(":")

    # Remove whitespace
    name = name.strip()
    signal_type = signal_type.strip()

    # Extract bitwidth from signal type
    match = re.match(type_pattern, signal_type)
    if match:
      bitwidth = int(match.group(1))
      extra_signals_dict[name] = bitwidth
    else:
      raise ValueError(f"Type {signal_type} of {name} is invalid")

  return extra_signals_dict

class VhdlScalarType:

  mlir_type: str
  # Note: VHDL only requires information on bitwidth and extra signals
  bitwidth: int
  extra_signals: dict[str, int] # key: name, value: bitwidth (todo)

  def __init__(self, mlir_type: str):
    """
    Constructor for VhdlScalarType.
    Parses an incoming MLIR type string.
    """
    self.mlir_type = mlir_type

    control_pattern = r"^!handshake\.control<(?:\[([^\]]*)\])?>$"
    channel_pattern = r"^!handshake\.channel<u?i(\d+)(?:, \[([^\]]*)\])?>$"

    match = re.match(control_pattern, mlir_type)
    if match:
      self.bitwidth = 0
      if match.group(1):
        self.extra_signals = parse_extra_signals(match.group(1))
      else:
        self.extra_signals = {}
      return

    match = re.match(channel_pattern, mlir_type)
    if match:
      self.bitwidth = int(match.group(1))
      if match.group(2):
        self.extra_signals = parse_extra_signals(match.group(2))
      else:
        self.extra_signals = {}
      return

    raise ValueError(f"Type {mlir_type} is invalid")

  def has_extra_signals(self):
    return bool(self.extra_signals)

  def is_channel(self):
    return self.bitwidth > 0

def generate_extra_signal_ports(ports, extra_signals):
  return "    -- extra signal ports\n" + "\n".join([
    "\n".join([
      f"    {port}_{name} : {inout} std_logic_vector({bitwidth - 1} downto 0);"
      for name, bitwidth in extra_signals.items()
    ])
    for port, inout in ports
  ])

# For concat-type signal managers (e.g., tehb, fork)
class ExtraSignalMapping:
  mapping: list[tuple[str, tuple[int, int]]]
  total_bitwidth: int
  def __init__(self, offset: int = 0):
    self.mapping = []
    self.total_bitwidth = offset
  def add(self, name: str, bitwidth: int):
    self.mapping.append((name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
    self.total_bitwidth += bitwidth
  def has(self, name: str) -> bool:
    return name in [name for name, _ in self.mapping]
  def get(self, name: str):
    return self.mapping[[name for name, _ in self.mapping].index(name)]

def generate_ins_concat_exp(in_name: str, extra_signal_mapping: ExtraSignalMapping) -> str:
  """
  Generates the input signal concatenation expression.
  in_name: The name of the input signal. (e.g., "ins")
  extra_signal_mapping: An ExtraSignalMapping object.
  e.g., "ins_tag & ins_spec & ins"
  """
  return f"{generate_ins_concat_exp_dataless(in_name, extra_signal_mapping)} & {in_name}"

def generate_ins_concat_exp_dataless(in_name: str, extra_signal_mapping: ExtraSignalMapping) -> str:
  """
  Generates the input signal concatenation expression.
  in_name: The name of the input signal. (e.g., "ins")
  extra_signals: An ExtraSignalMapping object.
  e.g., "ins_tag & ins_spec"
  """
  return ' & '.join([
    in_name + "_" + name for name, _ in extra_signal_mapping.mapping
  ].reverse())

def generate_outs_concat_statement(out_name: str, out_inner_name: str, extra_signal_mapping: ExtraSignalMapping, bitwidth: int, indent=2) -> str:
  """
  Generates the output signal concatenation statement.
  out_name: The name of the output signal. (e.g., "outs")
  out_inner_name: The name of the inner output signal. (e.g., "outs_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  bitwidth: The bitwidth of the data signal.
  e.g., outs <= outs_inner(31 downto 0)
  outs_spec <= outs_inner(32 downto 32)
  outs_tag <= outs_inner(40 downto 33)
  """
  indent_str = " " * indent
  return f"{indent_str}{out_name} <= {out_inner_name}({bitwidth - 1} downto 0);\n" + \
    generate_outs_concat_statement_dataless(out_name, out_inner_name, extra_signal_mapping, indent)

def generate_outs_concat_statement_dataless(out_name: str, out_inner_name: str, extra_signal_mapping: ExtraSignalMapping, indent=2) -> str:
  """
  Generates the output signal concatenation statement.
  out_name: The name of the output signal. (e.g., "outs")
  out_inner_name: The name of the inner output signal. (e.g., "outs_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  bitwidth: The bitwidth of the data signal.
  e.g., outs_spec <= outs_inner(0 downto 0)
  outs_tag <= outs_inner(8 downto 1)
  """
  indent_str = " " * indent
  return "\n".join([
    f"{indent_str}{out_name}_{name} <= {out_inner_name}({msb} downto {lsb})" for name, (msb, lsb) in extra_signal_mapping.mapping
  ])

def generate_extra_signal_concat_logic(ins: tuple[str, str], outs: list[tuple[str, str]], bit_map: tuple[dict[str, tuple[int, int]]]) -> str:
  """
  Generates the logic for extra signal concatenation.
  ins: tuple specifying input signal. e.g., ("ins", "ins_inner")
  outs: tuple specifying output signal.
  bit_map: A bit map for the extra signals.
  """
  ins_logic = []
  for name, (msb, lsb) in bit_map[0].items():
    ins_logic.append(f"  {name} <= ins_inner({msb} downto {lsb})")
  outs_logic = []
  for name, (msb, lsb) in bit_map[0].items():
    outs_logic.append(f"  outs_inner({msb} downto {lsb}) <= {name}")
  return "\n".join(ins_logic + outs_logic)
