-- customTypes.vhdl
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

package customTypes is
  -- Define the new type name: data_array
  type data_array is array (integer range <>) of std_logic_vector (natural range <>);
end customTypes;
