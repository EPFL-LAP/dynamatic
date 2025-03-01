library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tehb_chain is
  generic (
    DATA_TYPE : integer;
    NUM_SLOTS : integer
  );
  port (
    clk, rst   : in  std_logic;
    -- Input channel
    ins        : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid  : in  std_logic;
    ins_ready  : out std_logic;
    -- Output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of tehb_chain is
  type data_array  is array (0 to NUM_SLOTS) of std_logic_vector(DATA_TYPE - 1 downto 0);
  type valid_array is array (0 to NUM_SLOTS) of std_logic;
  type ready_array is array (0 to NUM_SLOTS) of std_logic;

  signal data_signals  : data_array;
  signal valid_signals : valid_array;
  signal ready_signals : ready_array;
begin
  data_signals(0)  <= ins;
  valid_signals(0) <= ins_valid;
  ins_ready        <= ready_signals(0);

  outs              <= data_signals(NUM_SLOTS);
  outs_valid        <= valid_signals(NUM_SLOTS);
  ready_signals(NUM_SLOTS) <= outs_ready;

  gen_tehb_chain: for i in 0 to NUM_SLOTS - 1 generate
    tehb_inst: entity work.tehb(arch)
      generic map (
        DATA_TYPE => DATA_TYPE
      )
      port map (
        clk        => clk,
        rst        => rst,
        ins        => data_signals(i),
        ins_valid  => valid_signals(i),
        ins_ready  => ready_signals(i),
        outs       => data_signals(i+1),
        outs_valid => valid_signals(i+1),
        outs_ready => ready_signals(i+1)
      );
  end generate;
end architecture;