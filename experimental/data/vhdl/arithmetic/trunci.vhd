-------------------
--trunci
----------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity trunci is
    Generic (
     INPUT_BITWIDTH: integer; OUTPUT_BITWIDTH: integer
    );
    port (
      clk : IN STD_LOGIC;
      rst : IN STD_LOGIC;
      pValidArray : IN std_logic_vector(0 downto 0);
      nReadyArray : in std_logic_vector(0 downto 0);
      validArray : out std_logic_vector(0 downto 0);
      readyArray : OUT std_logic_vector(0 downto 0);
      dataInArray : in data_array (0 downto 0)(INPUT_BITWIDTH-1 downto 0); 
      dataOutArray : out data_array (0 downto 0)(OUTPUT_BITWIDTH-1 downto 0));
end entity;
    
architecture arch of trunci is

    component my_trunc is
        port (
            ap_clk : IN STD_LOGIC;
            ap_rst : IN STD_LOGIC;
            ap_start : IN STD_LOGIC;
            ap_done : OUT STD_LOGIC;
            ap_idle : OUT STD_LOGIC;
            ap_ready : OUT STD_LOGIC;
            din : IN STD_LOGIC_VECTOR (31 downto 0);
            ap_return : OUT STD_LOGIC_VECTOR (31 downto 0) 
        );
    end component;

    signal idle : std_logic;
    signal component_ready : std_logic;

begin 

    my_trunc_U1 : component my_trunc
    port map(
        ap_clk => clk,
        ap_rst => rst,
        ap_start => pValidArray(0),
        ap_done => validArray(0),
        ap_idle => idle,
        ap_ready => component_ready,
        din => dataInArray(0),
        ap_return => dataOutArray(0) 
    );

    readyArray(0) <= idle and nReadyArray(0);
        
end architecture;