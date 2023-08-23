library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity source is

  generic (
    BITWIDTH : integer
  );

  port (
    clk, rst : in std_logic;
    valid    : out std_logic;
    nReady   : in std_logic
  );
end source;

architecture arch of source is

begin

  valid <= '1';

end arch;
