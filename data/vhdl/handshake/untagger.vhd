library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity un_tagger is generic(
    SIZE : integer ;
    DATA_SIZE_IN: integer; 
    DATA_SIZE_OUT: integer; 
    TOTAL_TAG_SIZE: integer; 
    TAG_SIZE: integer; 
    TAG_OFFSET: integer
);
port(
        clk, rst      : in  std_logic;
        pValidArray : in std_logic_vector(SIZE - 1 downto 0);

        nReadyArray : in std_logic_vector(SIZE downto 0);  -- this signal and the one after include the extra output of the UNTAGGER that carries the freed up tag
        validArray : out std_logic_vector(SIZE downto 0);

        readyArray : out std_logic_vector(SIZE - 1 downto 0);

        dataInArray   : in  data_array(SIZE - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(SIZE - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);

        freeTag_data : out std_logic_vector(TAG_SIZE-1 downto 0);

        tagInArray : in data_array (SIZE - 1 downto 0)(TOTAL_TAG_SIZE-1 downto 0) 
        );
end un_tagger;

architecture arch of un_tagger is

signal join_valid : std_logic;
signal join_nReady : std_logic;
constant all_one : std_logic_vector(SIZE-1 downto 0) := (others => '1');

signal join_readyArray : std_logic_vector(SIZE downto 0);

constant tag_idx_lower : integer := TAG_SIZE * TAG_OFFSET; 
constant tag_idx_upper : integer := tag_idx_lower + TAG_SIZE - 1; 

begin
    
    j : entity work.join(arch) generic map(SIZE)
                port map(   pValidArray,
                            join_nReady, --nReadyArray(0),
                            join_valid,
                            readyArray);

    dataOutArray <= dataInArray;

    freeTag_data <= tagInArray(0)(tag_idx_upper downto tag_idx_lower);  -- take the tag of any of the inputs; they are all guaranteed to be the same

    process(join_valid)
    begin
        if(join_valid = '1') then 
            validArray <= (others => '1');
        else
            validArray <= (others => '0');
        end if;
    end process;

    process (nReadyArray)
        variable check : std_logic := '1';
    begin
        check := '1';
        for I in 0 to SIZE loop
            check := check and nReadyArray(I);
        end loop;
        if(check = '1') then
            join_nReady <= '1';
        else 
            join_nReady <= '0';
        end if;
    end process;

end architecture;