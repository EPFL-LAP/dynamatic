library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity cond_br is generic (BITWIDTH : integer);
port (
  clk, rst    : in std_logic;
  pValidArray : in std_logic_vector(1 downto 0);
  condition   : in std_logic;
  dataInArray : in std_logic_vector(BITWIDTH - 1 downto 0);
  -- dataOutArray
  outToShift : out std_logic_vector(BITWIDTH - 1 downto 0);
  outShiftBy : out std_logic_vector(BITWIDTH - 1 downto 0);
  --
  nReadyArray : in std_logic_vector(1 downto 0);   -- (br1, br0)
  validArray  : out std_logic_vector(1 downto 0);  -- (br1, br0)
  readyArray  : out std_logic_vector(1 downto 0)); -- (data, condition)
end cond_br;
architecture arch of cond_br is
  signal joinValid, brReady : std_logic;
  --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
begin

  j : entity work.join(arch) generic map(2)
    port map(
    (pValidArray(1), pValidArray(0)),
      brReady,
      joinValid,
      readyArray);
  cond_br : entity work.branchSimple(arch)
    port map(
      condition,
      joinValid,
      nReadyArray,
      validArray,
      brReady);

  process (dataInArray)
  begin
    outToShift <= dataInArray;
    outShiftBy <= dataInArray;
  end process;

end architecture;
