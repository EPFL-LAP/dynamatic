from typing import cast, TypedDict, NotRequired
from .types import Port, ArrayPort, Direction
from .internal_signal import generate_internal_signal_vector, generate_internal_signal_array


class ConcatInfo:
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

  def has(self, name: str) -> bool:
    return name in [name for name, _ in self.mapping]

  def get(self, name: str):
    return self.mapping[[name for name, _ in self.mapping].index(name)]


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
  return sum(extra_signals.values())


class ConcatPortConversion(TypedDict):
  original_name: str
  original_bitwidth: int
  inner_name: str
  array_size: NotRequired[int]


def get_default_port_conversion(port: Port) -> ConcatPortConversion:
  inner_name = f"{port['name']}_inner"

  port_conversion: ConcatPortConversion = {
      "original_name": port["name"],
      "original_bitwidth": port["bitwidth"],
      "inner_name": inner_name
  }

  if port.get("array", False):
    port = cast(ArrayPort, port)
    port_conversion["array_size"] = port["size"]

  return port_conversion


def generate_concat_signal_decl(port_conversion: ConcatPortConversion, extra_signals_bitwidth: int) -> str:
  array_size = port_conversion.get("array_size", 0)

  # Concatenated bitwidth
  full_bitwidth = extra_signals_bitwidth + port_conversion["original_bitwidth"]

  if array_size > 0:
    return generate_internal_signal_array(port_conversion["inner_name"], full_bitwidth, array_size)
  else:
    return generate_internal_signal_vector(port_conversion["inner_name"], full_bitwidth)


def generate_concat_signal_decls(port_conversions: list[ConcatPortConversion], extra_signals_bitwidth: int) -> str:
  """
  Declare signals for concatenated data and extra signals
  e.g., signal lhs_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
  """
  signal_decls: list[str] = []
  for port_conversion in port_conversions:
    signal_decls.append(generate_concat_signal_decl(
        port_conversion, extra_signals_bitwidth))

  return "\n".join(signal_decls).lstrip()


def generate_concat_signal_decls_from_ports(ports: list[Port], extra_signals_bitwidth: int) -> str:
  return generate_concat_signal_decls([
      get_default_port_conversion(port) for port in ports
  ], extra_signals_bitwidth)


def generate_concat_port_assignment(port_conversion: ConcatPortConversion, dir: Direction, concat_info: ConcatInfo):
  original_name = port_conversion["original_name"]
  original_bitwidth = port_conversion["original_bitwidth"]
  inner_name = port_conversion["inner_name"]
  array_size = port_conversion.get("array_size", 0)

  if dir == "in":
    if array_size > 0:
      concat_logic = []
      for i in range(array_size):
        # Include data if present
        if original_bitwidth > 0:
          concat_logic.append(
              f"  {inner_name}({i})({original_bitwidth} - 1 downto 0) <= {original_name}({i});")

        # Include all extra signals
        for signal_name, (msb, lsb) in concat_info.mapping:
          concat_logic.append(
              f"  {inner_name}({i})({msb + original_bitwidth} downto {lsb + original_bitwidth}) <= {original_name}_{i}_{signal_name};")

      return concat_logic
    else:
      concat_logic = []

      # Include data if present
      if original_bitwidth > 0:
        concat_logic.append(
            f"  {inner_name}({original_bitwidth} - 1 downto 0) <= {original_name};")

      # Include all extra signals
      for signal_name, (msb, lsb) in concat_info.mapping:
        concat_logic.append(
            f"  {inner_name}({msb + original_bitwidth} downto {lsb + original_bitwidth}) <= {original_name}_{signal_name};")

      return concat_logic
  else:
    if array_size > 0:
      concat_logic = []

      for i in range(array_size):
        # Extract data if present
        if original_bitwidth > 0:
          concat_logic.append(
              f"  {original_name}({i}) <= {inner_name}({i})({original_bitwidth} - 1 downto 0);")

        # Extract all extra signals
        for signal_name, (msb, lsb) in concat_info.mapping:
          concat_logic.append(
              f"  {original_name}_{i}_{signal_name} <= {inner_name}({i})({msb + original_bitwidth} downto {lsb + original_bitwidth});")

      return concat_logic
    else:
      concat_logic = []

      # Extract data if present
      if original_bitwidth > 0:
        concat_logic.append(
            f"  {original_name} <= {inner_name}({original_bitwidth} - 1 downto 0);")

      # Extract all extra signals
      for signal_name, (msb, lsb) in concat_info.mapping:
        concat_logic.append(
            f"  {original_name}_{signal_name} <= {inner_name}({msb + original_bitwidth} downto {lsb + original_bitwidth});")

      return concat_logic


def generate_concat_port_assignments(in_port_conversions: list[ConcatPortConversion], out_port_conversions: list[ConcatPortConversion], concat_info: ConcatInfo) -> str:
  # Unify input and output ports, and add direction
  unified_port_conversions: list[tuple[ConcatPortConversion, Direction]] = []
  for port_conversion in in_port_conversions:
    unified_port_conversions.append((port_conversion, "in"))
  for port_conversion in out_port_conversions:
    unified_port_conversions.append((port_conversion, "out"))

  concat_logic = []
  for port_conversion, dir in unified_port_conversions:
    concat_logic += generate_concat_port_assignment(
        port_conversion, dir, concat_info)

  return "\n".join(concat_logic).lstrip()


def generate_concat_port_assignments_from_ports(in_ports: list[Port], out_ports: list[Port], concat_info: ConcatInfo) -> str:
  return generate_concat_port_assignments(
      [get_default_port_conversion(port) for port in in_ports],
      [get_default_port_conversion(port) for port in out_ports],
      concat_info
  )
