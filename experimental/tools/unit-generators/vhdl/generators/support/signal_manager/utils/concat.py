from typing import cast, TypedDict, NotRequired
from .types import Port, ArrayPort, Direction
from .internal_signal import generate_internal_signal_vector, generate_internal_signal_array


# Holds the concatenation layout of extra signals.
# Tracks each signal's bit range and keeps the total combined bitwidth.
class ConcatLayout:
  # List of tuples of (extra_signal_name, (msb, lsb))
  # e.g., [("spec", (0, 0)), ("tag0", (8, 1))]
  mapping: list[tuple[str, tuple[int, int]]]
  total_bitwidth: int

  def __init__(self, extra_signals: dict[str, int]):
    self.mapping = []
    self.total_bitwidth = 0

    for name, bitwidth in extra_signals.items():
      self.add(name, bitwidth)

  def add(self, name: str, bitwidth: int):
    self.mapping.append(
        (name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
    self.total_bitwidth += bitwidth


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
  return sum(extra_signals.values())


# Describes how a port is transformed during concatenation.
# Maps the original port to its new name.
class ConcatPortConversion(TypedDict):
  original_name: str
  original_bitwidth: int
  concat_name: str
  # If port is `data_array`, this is the size of the array.
  array_size: NotRequired[int]


def get_default_concat_name(original_name: str) -> str:
  """Generate the default name used for the internal concatenated signal."""
  return f"{original_name}_inner"


def get_default_port_conversion(port: Port) -> ConcatPortConversion:
  """
  Convert a port definition to its ConcatPortConversion form.
  Adds the default internal name and array size if applicable.
  """
  port_conversion: ConcatPortConversion = {
      "original_name": port["name"],
      "original_bitwidth": port["bitwidth"],
      "concat_name": get_default_concat_name(port["name"])
  }

  # Include array size if the port is an array
  if port.get("array", False):
    port = cast(ArrayPort, port)
    port_conversion["array_size"] = port["size"]

  return port_conversion


def generate_concat_signal_decl(port_conversion: ConcatPortConversion, extra_signals_bitwidth: int) -> str:
  """
  Generate a signal declaration for a port with extra signals concatenated.
  e.g., signal dataIn_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
  """
  array_size = port_conversion.get("array_size", 0)
  full_bitwidth = extra_signals_bitwidth + port_conversion["original_bitwidth"]

  if array_size > 0:
    return generate_internal_signal_array(port_conversion["concat_name"], full_bitwidth, array_size)
  else:
    return generate_internal_signal_vector(port_conversion["concat_name"], full_bitwidth)


def generate_concat_signal_decls(port_conversions: list[ConcatPortConversion], extra_signals_bitwidth: int) -> list[str]:
  """
  Generate a list of signal declarations for all ports with extra signals concatenated.
  e.g.,
  signal dataIn1_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
  signal dataIn2_inner : std_logic_vector(9 - 1 downto 0); // 8 (data) + 1 (spec)
  """
  signal_decls: list[str] = []
  for port_conversion in port_conversions:
    signal_decls.append(generate_concat_signal_decl(
        port_conversion, extra_signals_bitwidth))

  return signal_decls


def generate_concat_signal_decls_from_ports(ports: list[Port], extra_signals_bitwidth: int) -> list[str]:
  """
  High-level helper: convert ports to port conversions and generate signal declarations.
  """
  return generate_concat_signal_decls([
      get_default_port_conversion(port) for port in ports
  ], extra_signals_bitwidth)


def generate_concat_in_port_assignment(port_conversion: ConcatPortConversion, concat_layout: ConcatLayout) -> list[str]:
  """
  Generate VHDL assignments to pack original input signals and extra signals
  into a single concatenated internal signal.
  """
  original_name = port_conversion["original_name"]
  original_bitwidth = port_conversion["original_bitwidth"]
  concat_name = port_conversion["concat_name"]
  array_size = port_conversion.get("array_size", 0)

  concat_logic = []

  if array_size == 0:
    # Include data if present
    if original_bitwidth > 0:
      concat_logic.append(
          f"{concat_name}({original_bitwidth} - 1 downto 0) <= {original_name};")

    # Include all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      concat_logic.append(
          f"{concat_name}({msb + original_bitwidth} downto {lsb + original_bitwidth}) <= {original_name}_{signal_name};")
  else:
    # Signal is an array
    for i in range(array_size):
      # Include data if present
      if original_bitwidth > 0:
        concat_logic.append(
            f"{concat_name}({i})({original_bitwidth} - 1 downto 0) <= {original_name}({i});")

      # Include all extra signals
      for signal_name, (msb, lsb) in concat_layout.mapping:
        concat_logic.append(
            f"{concat_name}({i})({msb + original_bitwidth} downto {lsb + original_bitwidth}) <= {original_name}_{i}_{signal_name};")

  return concat_logic


def generate_concat_out_port_assignment(port_conversion: ConcatPortConversion, concat_layout: ConcatLayout) -> list[str]:
  """
  Generate VHDL assignments to unpack data and extra signals
  from a concatenated internal signal to the original output ports.
  """
  original_name = port_conversion["original_name"]
  original_bitwidth = port_conversion["original_bitwidth"]
  concat_name = port_conversion["concat_name"]
  array_size = port_conversion.get("array_size", 0)

  concat_logic = []

  if array_size == 0:
    # Extract data if present
    if original_bitwidth > 0:
      concat_logic.append(
          f"{original_name} <= {concat_name}({original_bitwidth} - 1 downto 0);")

    # Extract all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      concat_logic.append(
          f"{original_name}_{signal_name} <= {concat_name}({msb + original_bitwidth} downto {lsb + original_bitwidth});")

  else:
    # Signal is an array
    for i in range(array_size):
      # Extract data if present
      if original_bitwidth > 0:
        concat_logic.append(
            f"{original_name}({i}) <= {concat_name}({i})({original_bitwidth} - 1 downto 0);")

      # Extract all extra signals
      for signal_name, (msb, lsb) in concat_layout.mapping:
        concat_logic.append(
            f"{original_name}_{i}_{signal_name} <= {concat_name}({i})({msb + original_bitwidth} downto {lsb + original_bitwidth});")

  return concat_logic


def generate_concat_port_assignments(in_port_conversions: list[ConcatPortConversion], out_port_conversions: list[ConcatPortConversion], concat_layout: ConcatLayout) -> list[str]:
  """
  Generate VHDL assignments for both input and output ports
  using the given ConcatLayout.
  """
  concat_logic = []
  for port_conversion in in_port_conversions:
    concat_logic += generate_concat_in_port_assignment(
        port_conversion, concat_layout)
  for port_conversion in out_port_conversions:
    concat_logic += generate_concat_out_port_assignment(
        port_conversion, concat_layout)

  return concat_logic


def generate_concat_port_assignments_from_ports(in_ports: list[Port], out_ports: list[Port], concat_layout: ConcatLayout) -> list[str]:
  """
  High-level helper: convert input/output ports to default conversions and
  generate the full list of concatenation assignments.
  """
  return generate_concat_port_assignments(
      [get_default_port_conversion(port) for port in in_ports],
      [get_default_port_conversion(port) for port in out_ports],
      concat_layout
  )
