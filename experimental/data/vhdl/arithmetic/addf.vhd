----------------------------------------------------------------------- 
-- float add, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity addf is
Generic (
BITWIDTH: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(BITWIDTH-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0));
end entity;

architecture arch of addf is

    -- Interface to Vivado component
    component array_RAM_fadd_32bkb is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 10;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
    begin 
  

        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(8)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    
        array_RAM_fadd_32ns_32ns_32_10_full_dsp_1_U1 :  component array_RAM_fadd_32bkb
        port map (
            clk   => clk,
            reset => rst,
            ce    => oehb_ready,
            din0  => dataInArray(0),
            din1  => dataInArray(1),
            dout  => dataOutArray(0));
    
end architecture;