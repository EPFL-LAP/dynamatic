library IEEE;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity nandN is
  generic (n : integer := 4);
  port (
    x   : in std_logic_vector(N - 1 downto 0);
    res : out std_logic);
end nandN;

architecture arch of nandn is
  signal dummy  : std_logic_vector(n - 1 downto 0);
  signal andRes : std_logic;
begin
  dummy  <= (others => '1');
  andRes <= '1' when x = dummy else
    '0';
  res <= not andRes;
end arch;
