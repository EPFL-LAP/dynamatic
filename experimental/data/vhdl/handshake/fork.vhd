library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity fork is generic (
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

end fork;

architecture arch of fork is
  -- wrapper signals (internals use "stop" signals instead of "ready" signals)
  signal forkStop   : std_logic;
  signal nStopArray : std_logic_vector(OUTPUTS - 1 downto 0);
  -- internal combinatorial signals
  signal blockStopArray    : std_logic_vector(OUTPUTS - 1 downto 0);
  signal anyBlockStop      : std_logic;
  signal pValidAndForkStop : std_logic;
begin

  --can't adapt the signals directly in port map
  wrapper : process (forkStop, nReadyArray)
  begin
    ready <= not forkStop;
    for i in 0 to OUTPUTS - 1 loop
      nStopArray(i) <= not nReadyArray(i);
    end loop;
  end process;

  genericOr : entity work.orN generic map (OUTPUTS)
    port map(blockStopArray, anyBlockStop);

  -- internal combinatorial signals
  forkStop          <= anyBlockStop;
  pValidAndForkStop <= pValid and forkStop;

  --generate blocks
  generateBlocks : for i in OUTPUTS - 1 downto 0 generate
    regblock : entity work.eagerFork_RegisterBLock(arch)
      port map(
        clk, rst,
        pValid, nStopArray(i),
        pValidAndForkStop,
        validArray(i), blockStopArray(i));

  end generate;

  process (dataInArray)
  begin
    for I in 0 to OUTPUTS - 1 loop
      dataOutArray(I) <= dataInArray;
    end loop;
  end process;
end architecture;
