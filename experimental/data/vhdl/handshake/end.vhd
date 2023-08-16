--------------------------------------------------------------  end
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity end_node is 
    generic(
        MEM_INPUTS    : integer;
        BITWIDTH  : integer
    );

port (
    clk, rst : in std_logic;  
    dataInArray : in data_array (0  downto 0)(BITWIDTH-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);
    ReadyArray : out std_logic_vector(0  downto 0);
    ValidArray : out std_logic_vector(0  downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0);
    eReadyArray : out std_logic_vector(MEM_INPUTS - 1 downto 0);
    eValidArray : in std_logic_vector(MEM_INPUTS - 1  downto 0) := (others => '1'));
end end_node;

architecture arch of end_node is
    signal allPValid : std_logic;
    signal nReady: STD_LOGIC;
    signal valid: std_logic;
    signal mem_valid: std_logic;
    signal joinValid:std_logic;
    signal joinReady   : std_logic_vector(1 downto 0);

begin
   
    -- process for the return data
    -- there may be multiple return points, check if any is valid and output its data
    process(pValidArray, dataInArray)
        variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
        variable tmp_valid_out : std_logic;

    begin
        tmp_data_out  := unsigned(dataInArray(0));
        tmp_valid_out := '0';
        for I in 0 downto 0 loop
            if (pValidArray(I) = '1') then
                tmp_data_out  := unsigned(dataInArray(I));
                tmp_valid_out := pValidArray(I);
            end if;
        end loop;
    dataOutArray(0)  <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid <= tmp_valid_out;
    end process;

    -- check if all mem controllers are done (and of all valids from memory)
    mem_and: entity work.andN(vanilla) generic map (MEM_INPUTS)
            port map (eValidArray, mem_valid);

    -- join for return data and memory--we exit only in case the first process gets
    -- a single valid and if the AND of all memories is set
    j : entity work.join(arch) generic map(2)
            port map(   (valid, mem_valid),
                        nReadyArray(0),
                        joinValid,
                        joinReady);

   
    -- valid to successor (set by join)
    validArray(0) <= joinValid;

    -- join sends ready to predecessors
    -- not needed for eReady (because memory never reads it)
    process(joinReady)
    begin
        for I in 0 to 0 loop
            readyArray(I) <= joinReady(1);
        end loop;
    end process;

end arch;