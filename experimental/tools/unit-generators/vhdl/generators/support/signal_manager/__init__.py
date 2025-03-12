from collections.abc import Callable
from generators.support.utils import extra_signal_default_values, ExtraSignalMapping


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]):
  in_ports = params["in_ports"]
  out_ports = params["out_ports"]
  type = params["type"]

  if type == "normal":
    extra_signals = params["extra_signals"]
    return _generate_normal_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "concat":
    extra_signals = params["extra_signals"]
    return _generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)


def _generate_entity(entity_name, in_ports, out_ports):
  port_decls = []

  # Generate input port declarations
  for in_port in in_ports:
    name = in_port["name"]
    bitwidth = in_port["bitwidth"]
    extra_signals = in_port.get("extra_signals", None)

    if bitwidth > 0:
      port_decls.append(
          f"    {name} : in std_logic_vector({bitwidth} - 1 downto 0)")

    port_decls.append(f"    {name}_valid : in std_logic")
    port_decls.append(f"    {name}_ready : out std_logic")

    # Generate extra signal port declarations for this input port
    for signal_name, signal_bitwidth in extra_signals.items():
      port_decls.append(
          f"    {name}_{signal_name} : in std_logic_vector({signal_bitwidth} - 1 downto 0)")

  # Generate output port declarations
  for out_port in out_ports:
    name = out_port["name"]
    bitwidth = out_port["bitwidth"]
    extra_signals = out_port.get("extra_signals", None)

    if bitwidth > 0:
      port_decls.append(
          f"    {name} : out std_logic_vector({bitwidth} - 1 downto 0)")

    port_decls.append(f"    {name}_valid : out std_logic")
    port_decls.append(f"    {name}_ready : in std_logic")

    # Generate extra signal port declarations for this output port
    for signal_name, signal_bitwidth in extra_signals.items():
      port_decls.append(
          f"    {name}_{signal_name} : out std_logic_vector({signal_bitwidth} - 1 downto 0)")

  return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of signal manager
entity {entity_name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
{";\n".join(port_decls)}
  );
end entity;
"""


def _generate_normal_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]):
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = _generate_entity(name, in_ports, out_ports)

  # Generate extra signal expressions for each extra signal
  # We assume that all extra signals are ORed currently
  # e.g., {"spec": "lhs_spec or rhs_spec", "tag0": "lhs_tag0 or rhs_tag0"}
  extra_signal_exps: dict[str, str] = {}
  for signal_name in extra_signals:
    in_extra_signals = []

    if not in_ports:
      extra_signal_exps[signal_name] = extra_signal_default_values[signal_name]
    else:
      # Collect extra signals from all input ports
      for in_port in in_ports:
        port_name = in_port["name"]
        in_extra_signals.append(f"{port_name}_{signal_name}")

      extra_signal_exps[signal_name] = f" or ".join(in_extra_signals)

  # Generate extra signal assignments for each output port and extra signal,
  # based on the extra signal expressions
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []
  for out_port in out_ports:
    port_name = out_port["name"]

    for signal_name in extra_signals:
      extra_signal_assignments.append(
          f"  {port_name}_{signal_name} <= {extra_signal_exps[signal_name]};")

  # Port forwarding for inner entity
  ports = []
  for port in in_ports + out_ports:
    port_name = port["name"]
    bitwidth = port["bitwidth"]

    if bitwidth > 0:
      ports.append(f"      {port_name} => {port_name}")

    ports.append(f"      {port_name}_valid => {port_name}_valid")
    ports.append(f"      {port_name}_ready => {port_name}_ready")

  architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
begin

{"\n".join(extra_signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{",\n".join(ports)}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]):
  inner_name = f"{name}_inner"

  entity = _generate_entity(name, in_ports, out_ports)

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  inner = generate_inner(inner_name)

  inner_signal_decls = []
  for port in in_ports + out_ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]

    inner_signal_decls.append(
        f"  signal {port_name}_inner : std_logic_vector({extra_signals_bitwidth} + {port_bitwidth} - 1 downto 0);")

  concat_logic = []
  for port in in_ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]

    if port_bitwidth > 0:
      concat_logic.append(
          f"  {port_name}_inner({port_bitwidth} - 1 downto 0) <= {port_name};")

    for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
      concat_logic.append(
          f"  {port_name}_inner({msb} + {port_bitwidth} downto {lsb} + {port_bitwidth}) <= {port_name}_{signal_name};")

  for port in out_ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]

    if port_bitwidth > 0:
      concat_logic.append(
          f"  {port_name} <= {port_name}_inner({port_bitwidth} - 1 downto 0);")

    for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
      concat_logic.append(
          f"  {port_name}_{signal_name} <= {port_name}_inner({msb} + {port_bitwidth} downto {lsb} + {port_bitwidth});")

  # Port forwarding for inner entity
  ports = []
  for port in in_ports + out_ports:
    port_name = port["name"]

    ports.append(f"      {port_name} => {port_name}_inner")
    ports.append(f"      {port_name}_valid => {port_name}_valid")
    ports.append(f"      {port_name}_ready => {port_name}_ready")

  architecture = f"""
-- Architecture of signal manager (concat)
architecture arch of {name} is
  -- Concatenated data and extra signals
{"\n".join(inner_signal_decls)}
begin
{"\n".join(concat_logic)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{",\n".join(ports)}
    );
end architecture;
"""

  return inner + entity + architecture
