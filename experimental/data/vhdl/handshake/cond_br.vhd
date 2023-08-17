-------------------------------------------------------------  cond_br
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity branchSimple is port(
    condition,
    pValid : in std_logic;
    nReadyArray : in std_logic_vector(1 downto 0);  -- (cond_br1, cond_br0)
    validArray : out std_logic_vector(1 downto 0);
    ready : out std_logic);
end branchSimple;

---------------------------------------------------------------------
-- simple architecture
---------------------------------------------------------------------
architecture arch of branchSimple is
begin
    

    validArray(1) <= (not condition) and pValid;        
    validArray(0) <= condition and pValid;

    ready <= (nReadyArray(1) and not condition)
             or (nReadyArray(0) and condition);  

end arch;

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity cond_br is generic(BITWIDTH : integer);
port(
    clk, rst : in std_logic;
    pValidArray         : in std_logic_vector(1 downto 0);
    condition: in data_array (0 downto 0)(0 downto 0);
    dataInArray          : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    dataOutArray            : out data_array (1 downto 0)(BITWIDTH-1 downto 0);
    nReadyArray     : in std_logic_vector(1 downto 0);  -- (cond_br1, cond_br0)
    validArray      : out std_logic_vector(1 downto 0); -- (cond_br1, cond_br0)
    readyArray      : out std_logic_vector(1 downto 0));    -- (data, condition)
end cond_br;


architecture arch of cond_br is 
    signal joinValid, cond_brReady   : std_logic;
    --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
begin

    j : entity work.join(arch) generic map(2)
            port map(   (pValidArray(1), pValidArray(0)),
                        cond_brReady,
                        joinValid,
                        readyArray);

    cond_br : entity work.branchSimple(arch)
            port map(   condition(0)(0),
                        joinValid,
                        nReadyArray,
                        validArray,
                        cond_brReady);

    process(dataInArray)
    begin
        for I in 0 to 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process; 

end arch;