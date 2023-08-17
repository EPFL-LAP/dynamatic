--------------------------------------------------------------  merge
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge is

    generic(
        INPUTS        : integer;
        BITWIDTH  : integer
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end merge;

architecture arch of merge is
signal tehb_data_in  : std_logic_vector(BITWIDTH - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin

    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in  <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
        tehb_pvalid <= tmp_valid_out;

    end process;

    process(tehb_ready)
    begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
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

library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is

    generic(
        INPUTS        : integer;
        BITWIDTH  : integer;
    );
    port(
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0));
end merge_notehb;

architecture arch of merge_notehb is
signal tehb_data_in  : std_logic_vector(BITWIDTH - 1 downto 0);
signal tehb_pvalid : std_logic;
signal tehb_ready : std_logic;

begin

    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
        variable tmp_valid_out : std_logic;
    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in INPUTS - 1 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;

        tehb_data_in  <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
        tehb_pvalid <= tmp_valid_out;

    end process;

    process(tehb_ready)
    begin
        for I in 0 to INPUTS - 1 loop
            readyArray(I) <= tehb_ready;
        end loop;
    end process;

    tehb_ready <= nReadyArray(0);
    validArray(0) <= tehb_pvalid;
    dataOutArray(0) <= tehb_data_in;

end arch;