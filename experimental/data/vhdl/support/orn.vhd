library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity orN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end orN;

architecture vanilla of orN is
  signal dummy : std_logic_vector(n - 1 downto 0);
begin
  dummy <= (others => '0');
  res   <= '0' when x = dummy else
    '1';
end vanilla;
