library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity tagger is generic(
    SIZE : integer ; 
    DATA_SIZE_IN: integer; 
    DATA_SIZE_OUT: integer; 
    TOTAL_TAG_SIZE: integer; 
    TAG_SIZE: integer; 
    TAG_OFFSET: integer
);
port(
        clk, rst      : in  std_logic;
        pValidArray : in std_logic_vector(SIZE downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

        nReadyArray : in std_logic_vector(SIZE - 1 downto 0);
        validArray : out std_logic_vector(SIZE - 1 downto 0);

        readyArray : out std_logic_vector(SIZE downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

        dataInArray   : in  data_array(SIZE - 1 downto 0)(DATA_SIZE_IN - 1 downto 0);
        dataOutArray  : out data_array(SIZE - 1 downto 0)(DATA_SIZE_OUT - 1 downto 0);

        freeTag_data : in std_logic_vector(TAG_SIZE-1 downto 0);

        tagOutArray : out data_array (SIZE - 1 downto 0)(TOTAL_TAG_SIZE-1 downto 0) 
        );
end tagger;

architecture arch of tagger is

signal join_valid : std_logic;
signal join_nReady : std_logic;
constant all_one : std_logic_vector(SIZE-1 downto 0) := (others => '1');

constant tag_idx_lower : integer := TAG_SIZE * TAG_OFFSET; 
constant tag_idx_upper : integer := tag_idx_lower + TAG_SIZE - 1; 

signal join_readyArray : std_logic_vector(SIZE downto 0);

signal fork_ready: STD_LOGIC_VECTOR (0 downto 0);
signal fork_useless_out : data_array(SIZE - 1 downto 0)(0 downto 0);

begin
    
    j : entity work.join(arch) generic map(SIZE + 1)
                port map(   pValidArray,
                            join_nReady,
                            join_valid,
                            readyArray);

    dataOutArray <= dataInArray;

    tagging_process : process (freeTag_data)
    begin
      for I in 0 to SIZE - 1 loop
        tagOutArray(I)(tag_idx_upper downto tag_idx_lower) <= freeTag_data;
      end loop;
    end process;

    join_nReady <= fork_ready(0); 

    f : entity work.fork(arch) generic map(1, SIZE, 1, 1)
            port map (
        --inputs
            clk => clk, 
            rst => rst,  
            pValidArray(0) => join_valid,
            dataInArray (0) => "1",
            nReadyArray => nReadyArray, 
        --outputs
            dataOutArray => fork_useless_out,
            readyArray => fork_ready,  
            validArray => validArray    
            );

end architecture;