----------------------------------------------------------------------- 
-- ret, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity d_return is
Generic (
  BITWIDTH: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);      
  pValidArray : in std_logic_vector(0 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(0 downto 0));
end entity;

architecture arch of d_return is

begin 

tehb: entity work.TEHB(arch) generic map (BITWIDTH)
        port map (
        --inputs
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => pValidArray(0), 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => readyArray(0),   
            dataInArray(0) => dataInArray(0),
            dataOutArray(0) => dataOutArray(0)
        );

end architecture;