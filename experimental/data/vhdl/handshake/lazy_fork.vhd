-------------------------------------------------------------------  lazy_fork
------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity lazy_fork is generic(OUTPUTS : integer; BITWIDTH : Integer);
port(   clk, rst    : in std_logic; -- the eager implementation uses registers
        dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
        pValidArray : in std_logic_vector(0 downto 0);
        readyArray : out std_logic_v ector(0 downto 0);
        dataOutArray : out data_array (OUTPUTS-1 downto 0)(BITWIDTH-1 downto 0); 
        nReadyArray : in std_logic_vector(OUTPUTS-1 downto 0);
        validArray  : out std_logic_vector(OUTPUTS-1 downto 0)
        );
        
end lazy_fork;

architecture arch of lazy_fork is
    signal allnReady : std_logic;
begin

genericAnd : entity work.andn generic map (OUTPUTS)
    port map(nReadyArray, allnReady);
 
valids : process(pValidArray, nReadyArray, allnReady)
    begin
    for i in 0 to OUTPUTS-1 loop
        validArray(i) <= pValidArray(0) and allnReady;
    end loop;
    end process;
 
readyArray(0) <= allnReady;

process(dataInArray)
    begin
        for I in 0 to OUTPUTS - 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process;    

end arch;