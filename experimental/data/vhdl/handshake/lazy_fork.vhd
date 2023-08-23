library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity lazy_fork is generic (
  OUTPUTS  : integer;
  BITWIDTH : integer);
port (
  clk, rst     : in std_logic; -- the eager implementation uses registers
  dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
  pValid       : in std_logic;
  ready        : out std_logic;
  dataOutArray : out data_array (OUTPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
  nReadyArray  : in std_logic_vector(OUTPUTS - 1 downto 0);
  validArray   : out std_logic_vector(OUTPUTS - 1 downto 0)
);

end lazy_fork;

architecture arch of lazy_fork is
  signal allnReady : std_logic;
begin

  genericAnd : entity work.andn generic map (OUTPUTS)
    port map(nReadyArray, allnReady);

  valids : process (pValid, nReadyArray, allnReady)
  begin
    for i in 0 to OUTPUTS - 1 loop
      validArray(i) <= pValid and allnReady;
    end loop;
  end process;

  ready <= allnReady;

  process (dataInArray)
  begin
    for I in 0 to OUTPUTS - 1 loop
      dataOutArray(I) <= dataInArray;
    end loop;
  end process;

end arch;
