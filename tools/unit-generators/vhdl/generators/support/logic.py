def generate_and_n(name, params):
    size = params["size"]

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
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
-- Architecture of and_n
architecture arch of {name} is
  signal all_ones : std_logic_vector({size} - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;
"""

    return entity + architecture


def generate_or_n(name, params):
    size = params["size"]

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of or_n
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
-- Architecture of or_n
architecture arch of {name} is
  signal all_zeros : std_logic_vector({size} - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end architecture;
"""

    return entity + architecture
