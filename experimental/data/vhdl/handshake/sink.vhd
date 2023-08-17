--------------------------------------------------------------  sink
---------------------------------------------------------------------
library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity sink is

  Generic (
    BITWIDTH:integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    readyArray : out std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0)
  );
end sink;
 
architecture arch of sink is 

begin
 
readyArray(0) <= '1';

end arch;