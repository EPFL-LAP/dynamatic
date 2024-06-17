library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Simple request handling: it just takes whatever is more prioritized
-- request: a vector of requests encoded as 0/1
-- grant  : a one-hot encoded signal indicates which request has been granted
-- The convention is that MSBs are more prioritized than LSBs

entity bitscan is
  generic (
            size : unsigned
          );
  port (
    request : in std_logic_vector(size - 1 downto 0);
    grant : out std_logic_vector(size - 1 downto 0)
          );
begin
end bitscan;

architecture arch of bitscan is
begin
  p_bitscan : process(request)
  begin
    grant <= std_logic_vector(unsigned(request) and (not (unsigned(request) - 1)));
  end process p_bitscan;
end arch;
