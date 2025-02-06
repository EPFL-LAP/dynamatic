header = f"""
library ieee;
use ieee.std_logic_1164.all;
"""

def generate_and_n(name, size):
  entity = f"""
entity {name} is
  port (
    -- inputs
    ins : in std_logic_vector({size} - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;
"""

  architecture = f"""
architecture arch of {name} is
  signal all_ones : std_logic_vector({size} - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;
"""

  return header + entity + architecture
