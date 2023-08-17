----------------------------------------------------------------------- 
-- signed int division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity divsi is
Generic (
 BITWIDTH: integer
);
port(
 clk, rst : in std_logic; 
  dataInArray : in data_array (1 downto 0)(BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);      
  pValidArray : in std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of divsi is

    -- Interface to Vivado component
    component array_RAM_sdiv_32ns_32ns_32_36_1 is
        generic (
            ID : INTEGER;
            NUM_STAGE : INTEGER;
            din0_WIDTH : INTEGER;
            din1_WIDTH : INTEGER;
            dout_WIDTH : INTEGER);
        port (
            clk : IN STD_LOGIC;
            reset : IN STD_LOGIC;
            ce : IN STD_LOGIC;
            din0 : IN STD_LOGIC_VECTOR(din0_WIDTH - 1 DOWNTO 0);
            din1 : IN STD_LOGIC_VECTOR(din1_WIDTH - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(dout_WIDTH - 1 DOWNTO 0));
    end component;

    signal join_valid : STD_LOGIC;

begin 
    array_RAM_sdiv_32ns_32ns_32_36_1_U1 : component array_RAM_sdiv_32ns_32ns_32_36_1
    generic map (
        ID => 1,
        NUM_STAGE => 36,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 32)
    port map (
        clk => clk,
        reset => rst,
        ce => nReadyArray(0),
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        dout => dataOutArray(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray

    buff: entity work.delay_buffer(arch) 
    generic map(35)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;