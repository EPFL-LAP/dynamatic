library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

-- #NAME# = maxf, minf

entity #NAME# is
  generic (
    BITWIDTH : integer
  );
  port (
    clk         : in std_logic;
    rst         : in std_logic;
    pValidArray : in std_logic_vector(1 downto 0);
    nReady      : in std_logic;
    valid       : out std_logic;
    readyArray  : out std_logic_vector(1 downto 0);
    --dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0));
end entity;

architecture arch of #NAME# is

  component my_#NAME# is
    port (
      ap_clk    : in std_logic;
      ap_rst    : in std_logic;
      a         : in std_logic_vector (31 downto 0);
      b         : in std_logic_vector (31 downto 0);
      ap_return : out std_logic_vector (31 downto 0));
  end component;

  signal join_valid : std_logic;

begin

  my_trunc_U1 : component my_#NAME#
    port map(
      ap_clk    => clk,
      ap_rst    => rst,
      a         => inToShift,
      b         => inShiftBy,
      ap_return => dataOutArray
    );

    join_write_temp : entity work.join(arch) generic map(2)
      port map(
        pValidArray, --pValidArray
        nReady,      --nready                    
        join_valid,  --valid          
        readyArray); --readyarray 

    buff : entity work.delay_buffer(arch)
      generic map(1)
      port map(
        clk,
        rst,
        join_valid,
        nReady,
        valid);
  end architecture;
