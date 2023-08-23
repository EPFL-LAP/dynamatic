library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- #NAME# = divsi, divui, remsi, remui
entity #NAME# is
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

architecture arch of #NAME# is

  -- Interface to Vivado component
  component array_RAM_#NAME#_32ns_32ns_32_36_1 is
    generic (
      ID         : integer;
      NUM_STAGE  : integer;
      din0_WIDTH : integer;
      din1_WIDTH : integer;
      dout_WIDTH : integer);
    port (
      clk   : in std_logic;
      reset : in std_logic;
      ce    : in std_logic;
      din0  : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0));
  end component;

  signal join_valid : std_logic;

begin
  array_RAM_sdiv_32ns_32ns_32_36_1_U1 : component array_RAM_#NAME#_32ns_32ns_32_36_1
    generic map(
      ID         => 1,
      NUM_STAGE  => 36,
      din0_WIDTH => 32,
      din1_WIDTH => 32,
      dout_WIDTH => 32)
    port map(
      clk   => clk,
      reset => rst,
      ce    => nReady,
      din0  => inToShift,
      din1  => inShiftBy,
      dout  => dataOutArray);

    join_write_temp : entity work.join(arch) generic map(2)
      port map(
        pValidArray, --pValidArray
        nReady,      --nready                    
        join_valid,  --valid          
        readyArray); --readyarray

    buff : entity work.delay_buffer(arch)
      generic map(35)
      port map(
        clk,
        rst,
        join_valid,
        nReady,
        valid);
  end architecture;
