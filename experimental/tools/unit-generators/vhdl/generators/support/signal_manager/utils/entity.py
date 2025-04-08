from typing import cast
from .types import Port, ArrayPort, Direction


def generate_port_decl(port: Port, dir: Direction) -> list[str]:
  port_decls: list[str] = []

  ready_dir = "out" if dir == "in" else "in"
  name = port["name"]
  bitwidth = port["bitwidth"]
  extra_signals = port.get("extra_signals", {})
  port_array = port.get("array", False)

  if not port_array:
    # Usual case

    # Generate data signal if present
    if bitwidth > 0:
      port_decls.append(
          f"    {name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

    port_decls.append(f"    {name}_valid : {dir} std_logic")
    port_decls.append(f"    {name}_ready : {ready_dir} std_logic")

    # Generate extra signals for this input port
    for signal_name, signal_bitwidth in extra_signals.items():
      port_decls.append(
          f"    {name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
  else:
    # Port is array port
    port = cast(ArrayPort, port)
    size = port["size"]

    # Generate data_array signal declarations for 2d input port with bitwidth > 0
    if bitwidth > 0:
      port_decls.append(
          f"    {name} : {dir} data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0)")

    # Use std_logic_vector for valid/ready of 2d input port
    port_decls.append(
        f"    {name}_valid : {dir} std_logic_vector({size} - 1 downto 0)")
    port_decls.append(
        f"    {name}_ready : {ready_dir} std_logic_vector({size} - 1 downto 0)")

    # Use extra_signals_list if available to handle per-port extra signals
    use_extra_signals_list = "extra_signals_list" in port

    # Generate extra signal declarations for each item in the 2d input port
    for i in range(size):
      if use_extra_signals_list:
        # Use different extra signals for different ports
        current_extra_signals = port["extra_signals_list"][i]
      else:
        # Use the same extra signals for all items
        current_extra_signals = extra_signals

      # The netlist generator declares extra signals independently for each item,
      # in contrast to ready/valid signals.
      for signal_name, signal_bitwidth in current_extra_signals.items():
        port_decls.append(
            f"    {name}_{i}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")

  return port_decls


def generate_all_port_decls(in_ports: list[Port], out_ports: list[Port]) -> list[str]:
  # Unify input and output ports, and add direction
  unified_ports: list[tuple[Port, Direction]] = []
  for port in in_ports:
    unified_ports.append((port, "in"))
  for port in out_ports:
    unified_ports.append((port, "out"))

  port_decls = []
  # Add port declarations for each port
  for port, dir in unified_ports:
    port_decls += generate_port_decl(port, dir)

  return port_decls


def generate_entity_from_port_decls(entity_name: str, port_decls: list[str]) -> str:
  port_decls_str = ";\n".join(port_decls).lstrip()

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
  Generate entity for signal manager, based on input and output ports
  """

  port_decls = generate_all_port_decls(in_ports, out_ports)
  return generate_entity_from_port_decls(entity_name, port_decls)
