from .types import Port, Direction


def generate_port_decl(port: Port, dir: Direction) -> list[str]:
  """
  Generate VHDL port declarations for a given port and direction.
  Handles both scalar and array ports, including data, valid/ready,
  and extra signals.
  """

  port_decls: list[str] = []

  ready_dir = "out" if dir == "in" else "in"
  name = port["name"]
  bitwidth = port["bitwidth"]
  extra_signals = port.get("extra_signals", {})
  array_size = port.get("size", 0)

  if array_size == 0:
    # Scalar port

    if bitwidth > 0:
      # Declare data signal if present
      port_decls.append(
          f"{name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

    # Declare handshake signals
    port_decls.append(f"{name}_valid : {dir} std_logic")
    port_decls.append(f"{name}_ready : {ready_dir} std_logic")

    # Declare extra signals
    for signal_name, signal_bitwidth in extra_signals.items():
      port_decls.append(
          f"{name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
  else:
    # Array port

    if bitwidth > 0:
      # Declare 2D data array
      port_decls.append(
          f"{name} : {dir} data_array({array_size} - 1 downto 0)({bitwidth} - 1 downto 0)")

    # Handshake signals as 1D vector
    port_decls.append(
        f"{name}_valid : {dir} std_logic_vector({array_size} - 1 downto 0)")
    port_decls.append(
        f"{name}_ready : {ready_dir} std_logic_vector({array_size} - 1 downto 0)")

    # Check for per-index extra signal customization
    use_extra_signals_list = "extra_signals_list" in port

    for i in range(array_size):
      current_extra_signals = (
          port["extra_signals_list"][i] if use_extra_signals_list
          else extra_signals
      )

      # Declare per-item extra signals independently
      for signal_name, signal_bitwidth in current_extra_signals.items():
        port_decls.append(
            f"{name}_{i}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")

  return port_decls


def generate_all_port_decls(in_ports: list[Port], out_ports: list[Port]) -> list[str]:
  """
  Generate VHDL declarations for all input and output ports,
  combining both directions and delegating to `generate_port_decl`.
  """
  unified_ports: list[tuple[Port, Direction]] = []
  for port in in_ports:
    unified_ports.append((port, "in"))
  for port in out_ports:
    unified_ports.append((port, "out"))

  port_decls = []
  for port, dir in unified_ports:
    port_decls += generate_port_decl(port, dir)

  return port_decls


def generate_entity_from_port_decls(entity_name: str, port_decls: list[str]) -> str:
  """
  Generate full VHDL entity definition from a given entity name
  and list of port declarations.
  """
  port_decls_str = ";\n    ".join(port_decls)

  return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity {entity_name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
    {port_decls_str}
  );
end entity;
"""


def generate_entity(entity_name: str, in_ports: list[Port], out_ports: list[Port]) -> str:
  """
  High-level generator for a signal manager entity.
  Combines input and output port definitions and emits
  a full VHDL entity declaration.
  """

  port_decls = generate_all_port_decls(in_ports, out_ports)
  return generate_entity_from_port_decls(entity_name, port_decls)
