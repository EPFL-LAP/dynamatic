library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity constant is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    --dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    ready        : out std_logic;
    valid        : out std_logic;
    nReady       : in std_logic;
    pValid       : in std_logic);
end constant;

architecture arch of constant is
begin
  dataOutArray <= #CST_VALUE#;
  valid        <= pValid;
  ready        <= nReady;
end architecture;
