-------------------------------------------------------------  br
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;

entity branchSimple is port(
    condition,
    pValid : in std_logic;
    nReadyArray : in std_logic_vector(1 downto 0);  -- (br1, br0)
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
USE work.customTypes.all;

entity br is generic(BITWIDTH : integer);
port(
    clk, rst : in std_logic;
    pValidArray         : in std_logic_vector(1 downto 0);
    condition: in data_array (0 downto 0)(0 downto 0);
    dataInArray          : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    dataOutArray            : out data_array (1 downto 0)(BITWIDTH-1 downto 0);
    nReadyArray     : in std_logic_vector(1 downto 0);  -- (br1, br0)
    validArray      : out std_logic_vector(1 downto 0); -- (br1, br0)
    readyArray      : out std_logic_vector(1 downto 0));    -- (data, condition)
end br;


architecture arch of br is 
    signal joinValid, brReady   : std_logic;
    --signal dataOut0, dataOut1 : std_logic_vector(31 downto 0);
begin

    j : entity work.join(arch) generic map(2)
            port map(   (pValidArray(1), pValidArray(0)),
                        brReady,
                        joinValid,
                        readyArray);

    br : entity work.branchSimple(arch)
            port map(   condition(0)(0),
                        joinValid,
                        nReadyArray,
                        validArray,
                        brReady);

    process(dataInArray)
    begin
        for I in 0 to 1 loop
            dataOutArray(I) <= dataInArray(0);
        end loop;  
    end process; 

end arch;