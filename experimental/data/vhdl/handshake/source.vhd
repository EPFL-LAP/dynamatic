--------------------------------------------------------------  source
----------------------------------------------------------------------
library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity source is

  Generic (
    BITWIDTH:integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0)
  );
end source;



architecture arch of source is 

begin
 
validArray(0) <= '1';
