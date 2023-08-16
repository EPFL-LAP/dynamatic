-----------------------------------------------------------------------
-- sext, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity extsi is
Generic (
  INPUT_BITWIDTH: integer; OUTPUT_BITWIDTH: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(INPUT_BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(OUTPUT_BITWIDTH-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of extsi is

    signal join_valid : STD_LOGIC;

begin 

    dataOutArray(0)<= std_logic_vector(IEEE.numeric_std.resize(signed(dataInArray(0)),OUTPUT_BITWIDTH));
    validArray <= pValidArray;
    readyArray(0) <= not pValidArray(0) or (pValidArray(0) and nReadyArray(0));

end architecture;