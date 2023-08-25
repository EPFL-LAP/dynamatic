library ieee;
use ieee.std_logic_1164.all;

entity branchSimple is port (
  condition,
  pValid      : in std_logic;
  nReadyArray : in std_logic_vector(1 downto 0);
  validArray  : out std_logic_vector(1 downto 0);
  ready       : out std_logic);
end branchSimple;

architecture arch of branchSimple is
begin
  validArray(1) <= (not condition) and pValid;
  validArray(0) <= condition and pValid;

  ready <= (nReadyArray(1) and not condition)
    or (nReadyArray(0) and condition);

end arch;
