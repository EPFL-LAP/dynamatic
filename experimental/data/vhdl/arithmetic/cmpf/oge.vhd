-----------------------------------------------------------------------
-- cmpf oge, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity cmpf_oge is
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
    
architecture arch of cmpf_oge is

    --Interface to vivado component
    component array_RAM_cmpf_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00011";

begin 

    --TODO check with lana
    dataOutArray(0)(BITWIDTH - 1 downto 1) <= (others => '0');


    array_RAM_cmpf_32ns_32ns_1_2_1_u1 : component array_RAM_cmpf_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));


end architecture;