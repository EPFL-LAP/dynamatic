-----------------------------------------------------------------------
-- cmpi sle, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity cmpi_sle is
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

architecture arch of cmpi_sle is
-- slt: interprets the operands as signed values and yields true if op1 is less than op2.
    signal join_valid : STD_LOGIC;
    signal one: std_logic_vector (0 downto 0) := "1";
    signal zero: std_logic_vector (0 downto 0) := "0";

begin 

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    dataOutArray(0) <= one when (signed(dataInArray(0)) <= signed(dataInArray(1)) ) else zero;
    validArray(0) <= join_valid;

end architecture;