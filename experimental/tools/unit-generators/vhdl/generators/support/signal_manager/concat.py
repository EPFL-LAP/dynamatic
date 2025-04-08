from typing import cast
from collections.abc import Callable
from .types import Port, ArrayPort, Direction
from .mapping import generate_inner_port_mapping
from .entity import generate_entity
from .types import ExtraSignals


class ConcatenationInfo:
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


def generate_concat_signal_decl(signal_name: str, port: Port, extra_signals_bitwidth: int) -> str:
  port_bitwidth = port["bitwidth"]
  array = port.get("array", False)

  # Concatenated bitwidth
  full_bitwidth = extra_signals_bitwidth + port_bitwidth

  if array:
    port = cast(ArrayPort, port)
    port_size = port["size"]

    # Inner signal is data_array
    return f"  signal {signal_name} : data_array({port_size} - 1 downto 0)({full_bitwidth} - 1 downto 0);"
  else:
    return f"  signal {signal_name} : std_logic_vector({full_bitwidth} - 1 downto 0);"


def generate_concat_signal_decls(ports: list[Port], extra_signals_bitwidth: int, ignore=[]) -> str:
  """
  Declare signals for concatenated data and extra signals
  e.g., signal lhs_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
  """
  signal_decls: list[str] = []
  for port in ports:
    port_name = port["name"]

    # Ignore some ports
    if port_name in ignore:
      continue

    signal_decls.append(generate_concat_signal_decl(
        f"{port_name}_inner",
        port, extra_signals_bitwidth))

  return "\n".join(signal_decls).lstrip()


def generate_concat_port_assignment(port: Port, dir: Direction, concat_info: ConcatenationInfo):
  if dir == "in":
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]
    port_array = port.get("array", False)

    if port_array:
      port = cast(ArrayPort, port)
      port_size = port["size"]
      concat_logic = []
      for i in range(port_size):
        # Include data if present
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}_inner({i})({port_bitwidth} - 1 downto 0) <= {port_name}({i});")

        # Include all extra signals
        for signal_name, (msb, lsb) in concat_info.mapping:
          concat_logic.append(
              f"  {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{i}_{signal_name};")

      return concat_logic
    else:
      concat_logic = []

      # Include data if present
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name}_inner({port_bitwidth} - 1 downto 0) <= {port_name};")

      # Include all extra signals
      for signal_name, (msb, lsb) in concat_info.mapping:
        concat_logic.append(
            f"  {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{signal_name};")

      return concat_logic
  else:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]
    port_array = port.get("array", False)

    if port_array:
      port = cast(ArrayPort, port)
      port_size = port["size"]
      concat_logic = []

      for i in range(port_size):
        # Extract data if present
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}({i}) <= {port_name}_inner({i})({port_bitwidth} - 1 downto 0);")

        # Extract all extra signals
        for signal_name, (msb, lsb) in concat_info.mapping:
          concat_logic.append(
              f"  {port_name}_{i}_{signal_name} <= {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth});")

      return concat_logic
    else:
      concat_logic = []

      # Extract data if present
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name} <= {port_name}_inner({port_bitwidth} - 1 downto 0);")

      # Extract all extra signals
      for signal_name, (msb, lsb) in concat_info.mapping:
        concat_logic.append(
            f"  {port_name}_{signal_name} <= {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth});")

      return concat_logic


def generate_concat_port_assignments(in_ports: list[Port], out_ports: list[Port], concat_info: ConcatenationInfo, ignore=[]) -> str:
  # Unify input and output ports, and add direction
  unified_ports: list[tuple[Port, Direction]] = []
  for port in in_ports:
    unified_ports.append((port, "in"))
  for port in out_ports:
    unified_ports.append((port, "out"))

  concat_logic = []
  for port, dir in unified_ports:
    if port["name"] in ignore:
      continue
    concat_logic += generate_concat_port_assignment(port, dir, concat_info)

  return "\n".join(concat_logic).lstrip()


def generate_concat_mappings(ports: list[Port], extra_signals_bitwidth: int, handled_extra_signals: ExtraSignals, ignore_ports: list[str]) -> str:
  mappings = []
  for port in ports:
    port_name = port["name"]
    port_extra_signals = port.get("extra_signals", {})

    mapping_port = port.copy()
    mapping_port["bitwidth"] += extra_signals_bitwidth

    mapping_port_name = port_name if port_name in ignore_ports else f"{port_name}_inner"

    mapping_extra_signals = [signal_name
                             for signal_name in port_extra_signals
                             if signal_name not in handled_extra_signals]

    mappings += generate_inner_port_mapping(
        mapping_port, mapping_port_name, mapping_extra_signals)

  return ",\n".join(mappings).lstrip()


def generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Exclude specified ports for concatenation
  filtered_in_ports = [
      port for port in in_ports if not port["name"] in ignore_ports]
  filtered_out_ports = [
      port for port in out_ports if not port["name"] in ignore_ports]

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      filtered_in_ports + filtered_out_ports, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments(
      filtered_in_ports, filtered_out_ports, concat_info)

  # Port forwarding for the inner entity
  forwardings = generate_concat_mappings(
      in_ports + out_ports, extra_signals_bitwidth, extra_signals, ignore_ports)

  architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {forwardings}
    );
end architecture;
"""

  return inner + entity + architecture
