library ieee;
use ieee.std_logic_1164.all;

entity join is generic (SIZE : integer);
port (
  pValidArray : in std_logic_vector(SIZE - 1 downto 0);
  nReady      : in std_logic;
  valid       : out std_logic;
  readyArray  : out std_logic_vector(SIZE - 1 downto 0));
end join;

architecture arch of join is
  signal allPValid : std_logic;

begin

  allPValidAndGate : entity work.andN generic map(SIZE)
    port map(
      pValidArray,
      allPValid);

  valid <= allPValid;

  process (pValidArray, nReady)
    variable singlePValid : std_logic_vector(SIZE - 1 downto 0);
  begin
    for i in 0 to SIZE - 1 loop
      singlePValid(i) := '1';
      for j in 0 to SIZE - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and pValidArray(j));
        end if;
      end loop;
    end loop;
    for i in 0 to SIZE - 1 loop
      readyArray(i) <= (singlePValid(i) and nReady);
    end loop;
  end process;

end architecture;
