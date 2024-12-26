library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ofifo_dataless is
  generic (
    NUM_SLOTS : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of ofifo_dataless is
  signal tehb_valid, tehb_ready : std_logic;
  signal fifo_valid, fifo_ready : std_logic;
begin
  tehb : entity work.tehb_dataless(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.elastic_fifo_inner_dataless(arch) generic map (NUM_SLOTS)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
