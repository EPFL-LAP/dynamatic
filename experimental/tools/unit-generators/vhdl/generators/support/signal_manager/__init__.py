from collections.abc import Callable


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]):
  in_ports = params["in_ports"]
  out_ports = params["out_ports"]
  type = params["type"]

  inner = generate_inner(_get_inner_name(name))
  entity = _generate_entity(name, in_ports, out_ports)

  if type == "normal":
    extra_signals = params["extra_signals"]
    architecture = _generate_normal_architecture(
        name, in_ports, out_ports, extra_signals)

  return inner + entity + architecture


def _get_inner_name(name):
  return f"{name}_inner"


def _generate_entity(entity_name, in_ports, out_ports):
  port_decls = []

  # Generate input port declarations
  for in_port in in_ports:
    name = in_port["name"]
    bitwidth = in_port["bitwidth"]
    extra_signals = in_port.get("extra_signals", None)

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


def _generate_normal_architecture(arch_name, in_ports, out_ports, extra_signals):
  inner_name = _get_inner_name(arch_name)

  # Generate extra signal expressions for each extra signal
  # We assume that all extra signals are ORed currently
  # e.g., {"spec": "lhs_spec or rhs_spec", "tag0": "lhs_tag0 or rhs_tag0"}
  extra_signal_exps: dict[str, str] = {}
  for signal_name in extra_signals:
    in_extra_signals = []

    # Collect extra signals from all input ports
    for in_port in in_ports:
      name = in_port["name"]
      in_extra_signals.append(f"{name}_{signal_name}")

    extra_signal_exps[signal_name] = f" or ".join(in_extra_signals)

  # Generate extra signal assignments for each output port and extra signal,
  # based on the extra signal expressions
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []
  for out_port in out_ports:
    name = out_port["name"]

    for signal_name in extra_signals:
      extra_signal_assignments.append(
          f"  {name}_{signal_name} <= {extra_signal_exps[signal_name]};")

  # Port forwarding for inner entity
  ports = []
  for port in in_ports + out_ports:
    name = port["name"]
    ports.append(f"      {name} => {name}")
    ports.append(f"      {name}_valid => {name}_valid")
    ports.append(f"      {name}_ready => {name}_ready")

  return f"""
-- Architecture of signal manager (normal)
architecture arch of {arch_name} is
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
