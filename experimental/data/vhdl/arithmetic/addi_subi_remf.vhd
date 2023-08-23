library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

-- #NAME# = addi, subi, remf
-- #TYPEOP# = + (addi, remf), - (subi)
entity #NAME# is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    --dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValidArray  : in std_logic_vector(1 downto 0);
    nReady       : in std_logic;
    valid        : out std_logic;
    readyArray   : out std_logic_vector(1 downto 0));
end entity;

architecture arch of #NAME# is

  signal join_valid : std_logic;

begin

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
      pValidArray, --pValidArray
      nReady,      --nready                    
      join_valid,  --valid          
      readyArray); --readyarray 
  dataOutArray <= std_logic_vector(unsigned(inToShift) #TYPEOP# unsigned (inShiftBy));
  valid        <= join_valid;
end architecture;
