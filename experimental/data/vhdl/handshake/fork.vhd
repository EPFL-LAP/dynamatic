library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity fork is generic (
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
  outs_valid : out std_logic_vector(OUTPUTS - 1 downto 0)
);
end entity;

architecture arch of fork is
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

  genericOr : entity work.or_n generic map (OUTPUTS) port map(blockStopArray, anyBlockStop);
  forkStop          <= anyBlockStop;
  pValidAndForkStop <= ins_valid and forkStop;

  generateBlocks : for i in OUTPUTS - 1 downto 0 generate
    regblock : entity work.eager_fork_register_block(arch)
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

library ieee;
use ieee.std_logic_1164.all;

entity eager_fork_register_block is
  port (
    -- inputs
    clk, reset            : in std_logic;
    p_valid               : in std_logic;
    n_stop                : in std_logic;
    p_valid_and_fork_stop : in std_logic;
    -- outputs
    valid      : out std_logic;
    block_stop : out std_logic
  );
end entity;

architecture arch of eager_fork_register_block is
  signal reg_value, reg_in, block_stop_internal : std_logic;
begin
  block_stop_internal <= n_stop and reg_value;
  block_stop          <= block_stop_internal;
  reg_in              <= block_stop_internal or (not p_valid_and_fork_stop);
  valid               <= reg_value and p_valid;

  reg : process (clk, reset, reg_in)
  begin
    if (reset = '1') then
      reg_value <= '1'; --contains a "stop" signal - must be 1 at reset
    else
      if (rising_edge(clk)) then
        reg_value <= reg_in;
      end if;
    end if;
  end process reg;
end architecture;
