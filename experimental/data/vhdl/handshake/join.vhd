library ieee;
use ieee.std_logic_1164.all;

entity join is generic (SIZE : integer);
port (
  ins_valid  : in std_logic_vector(SIZE - 1 downto 0);
  outs_ready : in std_logic;
  outs_valid : out std_logic;
  ins_ready  : out std_logic_vector(SIZE - 1 downto 0));
end join;

architecture arch of join is
  signal allPValid : std_logic;

begin

  allPValidAndGate : entity work.andN generic map(SIZE)
    port map(
      ins_valid,
      allPValid);

  outs_valid <= allPValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(SIZE - 1 downto 0);
  begin
    for i in 0 to SIZE - 1 loop
      singlePValid(i) := '1';
      for j in 0 to SIZE - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to SIZE - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;
end architecture;
