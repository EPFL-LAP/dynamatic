library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity fork_node is generic (
  OUTPUTS  : integer;
  BITWIDTH : integer);
port (
  -- inputs
  ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
  ins_valid  : in std_logic;
  clk        : in std_logic;
  rst        : in std_logic;
  outs_ready : in std_logic_vector(OUTPUTS - 1 downto 0);
  -- outputs
  ins_ready  : out std_logic;
  outs       : out data_array (OUTPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
  outs_valid : out std_logic_vector(OUTPUTS - 1 downto 0));

end entity;

architecture arch of fork_node is
  signal forkStop          : std_logic;
  signal nStopArray        : std_logic_vector(OUTPUTS - 1 downto 0);
  signal blockStopArray    : std_logic_vector(OUTPUTS - 1 downto 0);
  signal anyBlockStop      : std_logic;
  signal pValidAndForkStop : std_logic;
begin
  wrapper : process (forkStop, outs_ready)
  begin
    ins_ready <= not forkStop;
    for i in 0 to OUTPUTS - 1 loop
      nStopArray(i) <= not outs_ready(i);
    end loop;
  end process;

  genericOr : entity work.orN generic map (OUTPUTS)
    port map(blockStopArray, anyBlockStop);
  forkStop          <= anyBlockStop;
  pValidAndForkStop <= ins_valid and forkStop;
  generateBlocks : for i in OUTPUTS - 1 downto 0 generate
    regblock : entity work.eagerFork_RegisterBLock(arch)
      port map(
        clk, rst,
        ins_valid, nStopArray(i),
        pValidAndForkStop,
        outs_valid(i), blockStopArray(i));

  end generate;

  process (ins)
  begin
    for I in 0 to OUTPUTS - 1 loop
      outs(I) <= ins;
    end loop;
  end process;
end architecture;
