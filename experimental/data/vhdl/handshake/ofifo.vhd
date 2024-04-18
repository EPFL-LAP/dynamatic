library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ofifo is
  generic (
    SIZE     : integer;
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(BITWIDTH downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(BITWIDTH downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of ofifo is
  signal tehb_valid, tehb_ready : std_logic;
  signal fifo_valid, fifo_ready : std_logic;
  signal tehb_dataOut, fifo_dataOut : std_logic_vector(BITWIDTH - 1 downto 0);

begin

  tehb : entity work.tehb(arch) generic map (BITWIDTH)
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

  fifo : entity work.elastic_fifo_inner(arch) generic map (SIZE, BITWIDTH)
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

  outs <= fifo_dataOut;
  outs_valid <= fifo_valid;
  ins_ready <= tehb_ready;

end arch;
