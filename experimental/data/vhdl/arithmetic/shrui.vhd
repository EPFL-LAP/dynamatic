-----------------------------------------------------------------------
-- lshr, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity shrui is
Generic (
BITWIDTH: integer
);
port(
  clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of shrui is

    signal join_valid : STD_LOGIC;

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= std_logic_vector(shift_right(unsigned(dataInArray(0)),to_integer(unsigned('0' & dataInArray(1)(BITWIDTH-2 downto 0)))));
    validArray(0) <= join_valid;


end architecture;