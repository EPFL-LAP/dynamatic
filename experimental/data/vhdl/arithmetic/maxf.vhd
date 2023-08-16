-------------------
--maxf
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity maxf is
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
    
architecture arch of maxf is

    component my_fmaxf is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            a : IN STD_LOGIC_VECTOR (31 downto 0);
            b : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) );
        end component;

    signal join_valid : std_logic;

begin 

    my_trunc_U1 : component my_fmaxf
    port map(
        ap_clk => clk,
        ap_rst => rst,
        a => dataInArray(0),
        b => dataInArray(1),
        ap_return => dataOutArray(0)
    );

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