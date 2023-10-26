library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity norN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end norN;

architecture arch of norN is
  signal dummy : std_logic_vector(n - 1 downto 0);
  signal orRes : std_logic;
begin
  dummy <= (others => '0');
  orRes <= '0' when x = dummy else
    '1';
  res <= not orRes;
end arch;
