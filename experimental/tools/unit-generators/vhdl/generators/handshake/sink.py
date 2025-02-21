from generators.support.utils import VhdlScalarType, generate_extra_signal_ports

def generate_sink(name, params):
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["ins"])

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of sink
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins       : in  std_logic_vector({data_type.bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([("ins", "in")], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]", extra_signal_ports)

  architecture = f"""
-- Architecture of sink
architecture arch of {name} is
begin
  ins_ready <= '1';
end architecture;
"""

  return entity + architecture
