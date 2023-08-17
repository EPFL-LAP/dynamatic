--------------------------------------------------------------  new_start
---------------------------------------------------------------------

library IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;
 use work.customTypes.all;
entity start_node is

  Generic (
    BITWIDTH:integer
  );
 
  Port ( 
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);
    readyArray : out std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0)
  );
end start_node;



architecture arch of start_node is 

signal set: STD_LOGIC;
signal start_internal:std_logic;
signal startBuff_readyArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_validArray : STD_LOGIC_VECTOR(0 downto 0);
signal startBuff_dataOutArray: data_array(0 downto 0)(BITWIDTH-1 downto 0);

begin
 
  process(clk, rst)
    begin

        if (rst=  '1')  then
        start_internal <= '0';
            set <= '0';

        elsif rising_edge(clk) then
            if (pValidArray(0) = '1' and set = '0') then
                start_internal<= '1';
                set <= '1';
            else 
                start_internal <= '0';
            end if;
        end if;
        
      
    end process;

startBuff: entity work.elasticBuffer(arch) generic map (BITWIDTH)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    dataInArray(0) => dataInArray(0),  ----dataInArray
    pValidArray(0) => start_internal,   --pValidArray
    nReadyArray(0) => nReadyArray(0),  --nReadyArray
--outputs
    dataOutArray => startBuff_dataOutArray,    ----dataOutArray
    readyArray => startBuff_readyArray,  --readyArray
    validArray => startBuff_validArray   --validArray
);

validArray(0) <= startBuff_validArray(0);
dataOutArray(0) <= startBuff_dataOutArray(0);
readyArray(0) <= startBuff_readyArray(0);

end arch;