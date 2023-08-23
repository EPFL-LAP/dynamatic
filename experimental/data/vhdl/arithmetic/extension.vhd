library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

-- #NAME# = extsi, extui

entity #NAME# is
  generic (
    INPUT_BITWIDTH  : integer;
    OUTPUT_BITWIDTH : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(INPUT_BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(OUTPUT_BITWIDTH - 1 downto 0);
    pValid       : in std_logic;
    nReady       : in std_logic;
    valid        : out std_logic;
    ready        : out std_logic);
end entity;

architecture arch of #NAME# is

  signal join_valid : std_logic;

begin

  dataOutArray <= std_logic_vector(IEEE.numeric_std.resize(signed(dataInArray), OUTPUT_BITWIDTH));
  valid        <= pValid;
  ready        <= not pValid or (pValid and nReady);

end architecture;
