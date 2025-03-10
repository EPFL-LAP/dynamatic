from generators.support.utils import generate_extra_signal_ports, extra_signal_default_values


def generate_source(name, params):
  extra_signals = params.get("extra_signals", None)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of source
entity {name} is
  port (
    clk, rst   : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- inputs
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports(
      [("outs", "out")], extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]", extra_signal_ports)

  architecture = f"""
-- Architecture of sink
architecture arch of {name} is
begin
  outs_valid <= '1';
  [EXTRA_SIGNAL_LOGIC]
end architecture;
"""

  extra_signal_assignments = []
  for signal_name in extra_signals:
    extra_signal_assignments.append(
        f"  outs_{signal_name} <= {extra_signal_default_values[signal_name]};")
  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]", "\n".join(extra_signal_assignments))

  return entity + architecture
