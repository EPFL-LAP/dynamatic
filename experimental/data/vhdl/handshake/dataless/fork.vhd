library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fork_dataless is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in  std_logic_vector(SIZE - 1 downto 0)
  );
end entity;

architecture arch of fork_dataless is
  signal forkStop          : std_logic;
  signal nStopArray        : std_logic_vector(SIZE - 1 downto 0);
  signal blockStopArray    : std_logic_vector(SIZE - 1 downto 0);
  signal anyBlockStop      : std_logic;
  signal pValidAndForkStop : std_logic;
begin
  wrapper : process (forkStop, outs_ready)
  begin
    ins_ready <= not forkStop;
    for i in 0 to SIZE - 1 loop
      nStopArray(i) <= not outs_ready(i);
    end loop;
  end process;

  genericOr : entity work.or_n generic map (SIZE) port map(blockStopArray, anyBlockStop);
  forkStop          <= anyBlockStop;
  pValidAndForkStop <= ins_valid and forkStop;

  generateBlocks : for i in SIZE - 1 downto 0 generate
    regblock : entity work.eager_fork_register_block(arch)
      port map(
        clk, rst,
        ins_valid, nStopArray(i),
        pValidAndForkStop,
        outs_valid(i), blockStopArray(i));
  end generate;

end architecture;
