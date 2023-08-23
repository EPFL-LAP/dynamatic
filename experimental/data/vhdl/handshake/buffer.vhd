library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity buffer is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    ready        : out std_logic;
    valid        : out std_logic;
    nReady       : in std_logic;
    pValid       : in std_logic);
end buffer;

architecture arch of buffer is

  signal tehb1_valid, tehb1_ready     : std_logic;
  signal oehb1_valid, oehb1_ready     : std_logic;
  signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(BITWIDTH - 1 downto 0);
begin

  tehb1 : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      --inputspValid
      clk    => clk,
      rst    => rst,
      pValid => pValid, -- real or speculatef condition (determined by merge1)
      nReady => oehb1_ready,
      valid  => tehb1_valid,
      --outputs
      ready        => tehb1_ready,
      dataInArray  => dataInArray,
      dataOutArray => tehb1_dataOut
    );

  oehb1 : entity work.OEHB(arch) generic map (BITWIDTH)
    port map(
      --inputspValid
      clk    => clk,
      rst    => rst,
      pValid => tehb1_valid, -- real or speculatef condition (determined by merge1)
      nReady => nReady,
      valid  => oehb1_valid,
      --outputs
      ready        => oehb1_ready,
      dataInArray  => tehb1_dataOut,
      dataOutArray => oehb1_dataOut
    );

  dataOutArray <= oehb1_dataOut;
  valid        <= oehb1_valid;
  ready        <= tehb1_ready;

end arch;
