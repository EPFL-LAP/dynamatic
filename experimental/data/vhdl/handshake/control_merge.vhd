library ieee;
use ieee.std_logic_1164.all;
USE work.customTypes.all;
entity control_merge is generic(
INPUTS : integer ; BITWIDTH: integer; COND_BITWIDTH:integer
);
port(
      clk, rst : in std_logic;    
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(1 downto 0);
        validArray : out std_logic_vector(1 downto 0);
        readyArray : out std_logic_vector(1 downto 0);
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        condition: out data_array(0 downto 0)(0 downto 0));
end control_merge;
architecture arch of control_merge is

signal phi_C1_readyArray : STD_LOGIC_VECTOR (1 downto 0);
signal phi_C1_validArray : STD_LOGIC_VECTOR (0 downto 0);
signal phi_C1_dataOutArray : data_array(0 downto 0)(0 downto 0);

signal fork_C1_readyArray : STD_LOGIC_VECTOR (0 downto 0);
signal fork_C1_dataOutArray : data_array(1 downto 0)(0 downto 0);
signal fork_C1_validArray : STD_LOGIC_VECTOR (1 downto 0);

signal oehb1_valid, oehb1_ready, index : STD_LOGIC;
signal oehb1_dataOut : std_logic_vector(BITWIDTH-1 downto 0);

begin


readyArray <= phi_C1_readyArray;

phi_C1: entity work.merge_notehb(arch) generic map (2, 1)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    pValidArray => pValidArray,    --pValidArray
    dataInArray (0) => "1",
    dataInArray (1) => "1",
    nReadyArray(0) => oehb1_ready,--outputs
    dataOutArray => phi_C1_dataOutArray,
    readyArray => phi_C1_readyArray,    --readyArray
    validArray => phi_C1_validArray --validArray
);


process(pValidArray)
begin
        if (pValidArray(0) = '1') then
            index <= '0';
        else
            index <= '1';
        end if;
end process;

oehb1: entity work.TEHB(arch) generic map (1)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => phi_C1_validArray(0), 
            nReadyArray(0) => fork_C1_readyArray(0),    
            validArray(0) => oehb1_valid, 
        --outputs
            readyArray(0) => oehb1_ready,   
            dataInArray(0)(0) => index,
            dataOutArray(0) => oehb1_dataOut
        );

fork_C1: entity work.fork(arch) generic map (2, 1)
port map (
--inputs
    clk => clk,  --clk
    rst => rst,  --rst
    pValidArray(0) => oehb1_valid, --pValidArray
    dataInArray (0) => "1",
    nReadyArray => nReadyArray, --nReadyArray
--outputs
    dataOutArray => fork_C1_dataOutArray,
    readyArray => fork_C1_readyArray,   --readyArray
    validArray => fork_C1_validArray    --validArray
);


validArray <= fork_C1_validArray;
condition(0) <= oehb1_dataOut;

end arch;