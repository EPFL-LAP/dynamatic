library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity end_sync_dataless is
  generic (
    NUM_MEMORIES : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic_vector(0 downto 0);
    ins_ready : out std_logic_vector(0 downto 0);
    -- memory input channels
    memDone_valid : in  std_logic_vector(NUM_MEMORIES - 1 downto 0) := (others => '1');
    memDone_ready : out std_logic_vector(NUM_MEMORIES - 1 downto 0);
    -- output channel
    outs_valid : out std_logic_vector(0 downto 0);
    outs_ready : in  std_logic_vector(0 downto 0)
  );
end entity;

architecture arch of end_sync_dataless is
  signal memReady, allMemDone : std_logic;
begin
  memDone_ready <= (others => '1');

  mem_and : entity work.and_n(arch) generic map(NUM_MEMORIES)
    port map(
      ins  => memDone_valid,
      outs => allMemDone
    );

  join_ins_mem : entity work.join(arch) generic map(2)
    port map(
      ins_valid(0) => ins_valid(0),
      ins_valid(1) => allMemDone,
      outs_ready   => outs_ready(0),
      outs_valid   => outs_valid(0),
      ins_ready(0) => ins_ready(0),
      ins_ready(1) => memReady
    );
end architecture;
