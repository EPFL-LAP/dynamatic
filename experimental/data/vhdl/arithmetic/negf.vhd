----------------------------------------------------------------------- 
-- float neg, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity negf is
Generic (
 BITWIDTH: integer
);
port (
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of negf is

    constant msb_mask : std_logic_vector(31 downto 0) := (31 => '1', others => '0');

begin 

    dataOutArray(0) <= dataInArray(0) xor msb_mask;
    validArray(0) <= pValidArray(0);
    readyArray(0) <= nReadyArray(0);


end architecture;