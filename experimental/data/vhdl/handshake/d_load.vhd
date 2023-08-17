library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;


entity d_load is generic(ADDR_BITWIDTH : Integer;  DATA_BITWIDTH : Integer);
port (
    rst: in std_logic;
    clk: in std_logic;

    --- interface to previous
    pValidArray : in std_logic_vector(1 downto 0);
    readyArray : out std_logic_vector(1 downto 0);
    dataInArray: in data_array (0 downto 0)(DATA_BITWIDTH -1 downto 0);
    input_addr: in std_logic_vector(ADDR_BITWIDTH -1 downto 0);

    ---interface to next
    nReadyArray : in std_logic_vector(1 downto 0);
    validArray : out std_logic_vector(1 downto 0);
    dataOutArray : out data_array (0 downto 0)(DATA_BITWIDTH-1 downto 0);
    output_addr: out std_logic_vector(ADDR_BITWIDTH -1 downto 0)
    );

end entity;

architecture arch of d_load is
    signal Buffer_1_readyArray_0 : std_logic;
    signal Buffer_1_validArray_0 : std_logic;
    signal Buffer_1_dataOutArray_0 : std_logic_vector(ADDR_BITWIDTH-1 downto 0);

    signal Buffer_2_readyArray_0 : std_logic;
    signal Buffer_2_validArray_0 : std_logic;
    signal Buffer_2_dataOutArray_0 : std_logic_vector(DATA_BITWIDTH-1 downto 0);

    signal addr_from_circuit: std_logic_vector(ADDR_BITWIDTH-1 downto 0);
    signal addr_from_circuit_valid: std_logic;
    signal addr_from_circuit_ready: std_logic;

    signal addr_to_lsq: std_logic_vector(ADDR_BITWIDTH-1 downto 0);
    signal addr_to_lsq_valid: std_logic;
    signal addr_to_lsq_ready: std_logic;

    signal data_from_lsq: std_logic_vector(DATA_BITWIDTH-1 downto 0);
    signal data_from_lsq_valid: std_logic;
    signal data_from_lsq_ready: std_logic;

    signal data_to_circuit: std_logic_vector(DATA_BITWIDTH-1 downto 0);
    signal data_to_circuit_valid: std_logic;
    signal data_to_circuit_ready: std_logic;

begin

    addr_from_circuit <= input_addr;
    addr_from_circuit_valid <= pValidArray(1);
    readyArray(1)  <= Buffer_1_readyArray_0;

    Buffer_1: entity work.TEHB(arch) generic map (ADDR_BITWIDTH)
    port map (
        clk => clk,
        rst => rst,
        dataInArray(0) => addr_from_circuit,
        pValidArray(0) => addr_from_circuit_valid,
        readyArray(0) => Buffer_1_readyArray_0,
        nReadyArray(0) => addr_to_lsq_ready,
        validArray(0) => Buffer_1_validArray_0,
        dataOutArray(0) => Buffer_1_dataOutArray_0
    );


    addr_to_lsq <= Buffer_1_dataOutArray_0;
    addr_to_lsq_valid <= Buffer_1_validArray_0;
    addr_to_lsq_ready <= nReadyArray(1);

    output_addr <= addr_to_lsq; -- address request goes to LSQ
    validArray(1) <= addr_to_lsq_valid;

    readyArray(0) <= data_from_lsq_ready;

    data_from_lsq <= dataInArray(0);
    data_from_lsq_valid <= pValidArray(0);
    data_from_lsq_ready <= Buffer_2_readyArray_0;

    dataOutArray(0) <= Buffer_2_dataOutArray_0; -- data from LSQ to load output
    validArray(0) <=  Buffer_2_validArray_0;

    Buffer_2: entity work.TEHB(arch) generic map (DATA_BITWIDTH)
    port map (
        clk => clk,
        rst => rst,
        dataInArray(0) => data_from_lsq,
        pValidArray(0) => data_from_lsq_valid,
        readyArray(0) => Buffer_2_readyArray_0,
        nReadyArray(0) => nReadyArray(0),
        validArray(0) => Buffer_2_validArray_0,
        dataOutArray(0) => Buffer_2_dataOutArray_0
    );

        
end architecture;