library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tfifo_dataless_with_tag is
  generic (
    NUM_SLOTS : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of tfifo_dataless_with_tag is
  signal ins_inner : std_logic_vector(0 downto 0);
  signal outs_inner : std_logic_vector(0 downto 0);
begin
  ins_inner(0) <= ins_spec_tag;
  outs_spec_tag <= outs_inner(0);
  tehb_inner : entity work.tfifo(arch) generic map(NUM_SLOTS, 1)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins_inner,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
