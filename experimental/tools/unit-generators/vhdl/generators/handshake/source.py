def generate_source(name, _):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of source
entity {name} is
  port (
    clk, rst   : in std_logic;
    -- inputs
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of sink
architecture arch of {name} is
begin
  outs_valid <= '1';
end arch;
"""

  return entity + architecture
