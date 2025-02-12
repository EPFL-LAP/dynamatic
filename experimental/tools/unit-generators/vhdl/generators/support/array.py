def generate_2d_array(name, first_dim, second_dim):
  return f"""
library ieee;
use ieee.std_logic_1164.all;

package {name} is
  -- Type of 2D array
  type {name} is array({first_dim} - 1 downto 0) of std_logic_vector({second_dim} - 1 downto 0);
end package;
"""