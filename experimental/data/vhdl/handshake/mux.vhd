----------------------------------------------------------------
-----------------------------MUX--------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity mux is
    generic(
        INPUTS        : integer;
        BITWIDTH  : integer;
        COND_BITWIDTH  : integer
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 2 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS -1 downto 0);
        nReadyArray      : in  std_logic_vector(0 downto 0);
        validArray      : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(INPUTS -1 downto 0);
        condition     : in  data_array(0 downto 0)(CONDITION - 1 downto 0)   ----(integer(ceil(log2(real(INPUTS)))) - 1 downto 0);
        
    );
end mux;

architecture arch of mux is

signal tehb_data_in  : std_logic_vector(BITWIDTH - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin
    process(dataInArray, pValidArray, nReadyArray, condition, tehb_ready)
        variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
        variable tmp_valid_out : std_logic;

        
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 2 downto 0 loop
            -- if (the condition refers the Ith data input, condition is valid, and the Ith input is valid), assign input data to output and set the output valid high
            if (unsigned(condition(0)) = to_unsigned(I,condition(0)'length) and pValidArray(0) = '1' and pValidArray(I+1) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := '1';
            end if;
            -- set the readyOutArray
            if ((unsigned(condition(0)) = to_unsigned(I,condition(0)'length) and pValidArray(0) = '1' and tehb_ready = '1' and pValidArray(I+1) = '1') or pValidArray(I+1) = '0') then
                readyArray(I+1) <= '1';
            else
                readyArray(I+1) <= '0';
            end if;
        end loop;
        -- set the condtionReady
        if (pValidArray(0) = '0' or (tmp_valid_out = '1' and tehb_ready = '1')) then
            readyArray(0) <= '1';
        else
            readyArray(0) <= '0';
        end if;
        --Assign dataout and validout
        tehb_data_in <= std_logic_vector(resize(tmp_data_out,BITWIDTH));
        tehb_pvalid     <= tmp_valid_out;
    end process;


    tehb1: entity work.TEHB(arch) generic map (BITWIDTH)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb_pvalid, 
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => tehb_ready,   
            dataInArray(0) => tehb_data_in,
            dataOutArray(0) => dataOutArray(0)
        );
end arch;