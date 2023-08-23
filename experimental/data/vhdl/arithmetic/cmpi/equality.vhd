library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- NAME = eq, ne
-- CONDTRUE = one, zero, 
-- CONDFALSE = zero, one
entity cmpi_#NAME# is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValidArray  : in std_logic_vector(1 downto 0);
    nReady       : in std_logic;
    valid        : out std_logic;
    readyArray   : out std_logic_vector(1 downto 0));
end entity;

architecture arch of cmpi_#NAME# is
  signal join_valid : std_logic;
  signal one        : std_logic := "1";
  signal zero       : std_logic := "0";

begin

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
      pValidArray, --pValidArray
      nReady,      --nready                    
      join_valid,  --valid          
      readyArray); --readyarray 

  dataOutArray <= #CONDTRUE# when (inToShift = inShiftBy) else
    #CONDFALSE#;
  valid <= join_valid;
end architecture;
