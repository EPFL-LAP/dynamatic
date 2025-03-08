from generators.support.utils import generate_extra_signal_ports


def generate_constant(name, params):
  bitwidth = params["bitwidth"]
  value = params["value"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_constant_signal_manager(name, value, bitwidth, extra_signals)
  else:
    return _generate_constant(name, value, bitwidth)


def _generate_constant(name, value, bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of constant
architecture arch of {name} is
begin
  outs       <= "{value}";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;
"""

  return entity + architecture


extra_signal_logic = {
    "spec": """
  outs_spec <= ctrl_spec;
"""
}


def _generate_constant_signal_manager(name, value, bitwidth, extra_signals):
  inner_name = f"{name}_inner"

  dependencies = _generate_constant(inner_name, value, bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
      ("ctrl", "in"),
      ("outs", "out")
  ], extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of constant signal manager
architecture arch of {name} is
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk       => clk,
      rst       => rst,
      ctrl_valid => ctrl_valid,
      ctrl_ready => ctrl_ready,
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  architecture = architecture.replace("  [EXTRA_SIGNAL_LOGIC]", "\n".join([
      extra_signal_logic[name] for name in extra_signals
  ]))

  return dependencies + entity + architecture
