library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity constant is
Generic (
  BITWIDTH: Integer 
);
port(
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);
    ReadyArray : out std_logic_vector(0 downto 0);
    ValidArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0));
end constant;

architecture arch of constant is
begin
dataOutArray <= dataInArray;
validArray <= pValidArray;
readyArray <= nReadyArray; 
end architecture;