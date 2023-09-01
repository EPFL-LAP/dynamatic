library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity andN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(n - 1 downto 0);
    res : out std_logic);
end andN;

architecture vanilla of andn is
  signal dummy : std_logic_vector(n - 1 downto 0);
begin
  dummy <= (others => '1');
  res   <= '1' when x = dummy else
    '0';
end vanilla;
