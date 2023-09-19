library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity buffer_seq is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end buffer_seq;

architecture arch of buffer_seq is

  signal tehb1_valid, tehb1_ready     : std_logic;
  signal oehb1_valid, oehb1_ready     : std_logic;
  signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  tehb1 : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => oehb1_ready,
      outs_valid => tehb1_valid,

      ins_ready => tehb1_ready,
      ins       => ins,
      outs      => tehb1_dataOut
    );

  oehb1 : entity work.OEHB(arch) generic map (BITWIDTH)
    port map(

      clk        => clk,
      rst        => rst,
      ins_valid  => tehb1_valid,
      outs_ready => outs_ready,
      outs_valid => oehb1_valid,

      ins_ready => oehb1_ready,
      ins       => tehb1_dataOut,
      outs      => oehb1_dataOut
    );

  outs       <= oehb1_dataOut;
  outs_valid <= oehb1_valid;
  ins_ready  <= tehb1_ready;

end arch;
