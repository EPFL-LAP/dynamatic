library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity d_return is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValid       : in std_logic;
    nReady       : in std_logic;
    valid        : out std_logic;
    ready        : out std_logic);
end entity;

architecture arch of d_return is

begin

  tehb : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      --inputs
      clk    => clk,
      rst    => rst,
      pValid => pValid,
      nReady => nReady,
      valid  => valid,
      --outputs
      ready        => ready,
      dataInArray  => dataInArray,
      dataOutArray => dataOutArray
    );

end architecture;
