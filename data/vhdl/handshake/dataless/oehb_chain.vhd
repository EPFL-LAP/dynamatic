library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity oehb_dataless_chain is
  generic (
    NUM_SLOTS : integer
  );
  port (
    clk, rst   : in  std_logic;
    -- Input channel
    ins_valid  : in  std_logic;
    ins_ready  : out std_logic;
    -- Output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of oehb_dataless_chain is
  type valid_array is array (0 to NUM_SLOTS) of std_logic;
  type ready_array is array (0 to NUM_SLOTS) of std_logic;

  signal valid_signals : valid_array;
  signal ready_signals : ready_array;
begin
  valid_signals(0) <= ins_valid;
  ins_ready        <= ready_signals(0);

  outs_valid           <= valid_signals(NUM_SLOTS);
  ready_signals(NUM_SLOTS) <= outs_ready;

  gen_oehb_dataless_chain: for i in 0 to NUM_SLOTS - 1 generate
    oehb_dataless_inst: entity work.oehb_dataless(arch)
      port map (
        clk        => clk,
        rst        => rst,
        ins_valid  => valid_signals(i),
        ins_ready  => ready_signals(i),
        outs_valid => valid_signals(i+1),
        outs_ready => ready_signals(i+1)
      );
  end generate;
end architecture;