library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity sink is

  generic (
    BITWIDTH : integer
  );

  port (
    clk, rst : in std_logic;
    ready    : out std_logic;
    pValid   : in std_logic
  );
end sink;

architecture arch of sink is

begin

  ready <= '1';

end arch;
