library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ofifo is
  generic (
    NUM_SLOTS  : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of ofifo is
  signal tehb_valid, tehb_ready     : std_logic;
  signal fifo_valid, fifo_ready     : std_logic;
  signal tehb_dataOut, fifo_dataOut : std_logic_vector(DATA_TYPE - 1 downto 0);
begin
  tehb : entity work.tehb(arch) generic map (DATA_TYPE)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs       => tehb_dataOut,
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.elastic_fifo_inner(arch) generic map (NUM_SLOTS, DATA_TYPE)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins        => tehb_dataOut,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs       => fifo_dataOut,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs       <= fifo_dataOut;
  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ofifo_with_tag is
  generic (
    NUM_SLOTS  : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of ofifo_with_tag is
  signal ins_inner : std_logic_vector(DATA_TYPE downto 0);
  signal outs_inner : std_logic_vector(DATA_TYPE downto 0);
begin
  ins_inner <= ins_spec_tag & ins;
  outs <= outs_inner(DATA_TYPE - 1 downto 0);
  outs_spec_tag <= outs_inner(DATA_TYPE);
  ofifo : entity work.ofifo(arch) generic map (NUM_SLOTS, DATA_TYPE + 1)
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
