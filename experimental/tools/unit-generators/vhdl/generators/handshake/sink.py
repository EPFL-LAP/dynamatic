def generate_sink(name, params):
  bitwidth = params["bitwidth"]

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of sink
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of sink
architecture arch of {name} is
begin
  ins_ready <= '1';
end architecture;
"""

  return entity + architecture
