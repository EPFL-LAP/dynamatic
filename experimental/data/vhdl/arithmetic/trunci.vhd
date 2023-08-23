library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity trunci is
  generic (
    INPUT_BITWIDTH  : integer;
    OUTPUT_BITWIDTH : integer
  );
  port (
    clk          : in std_logic;
    rst          : in std_logic;
    pValid       : in std_logic;
    nReady       : in std_logic;
    valid        : out std_logic;
    ready        : out std_logic;
    dataInArray  : in std_logic_vector(INPUT_BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(OUTPUT_BITWIDTH - 1 downto 0));
end entity;

architecture arch of trunci is

  component my_trunc is
    port (
      ap_clk    : in std_logic;
      ap_rst    : in std_logic;
      ap_start  : in std_logic;
      ap_done   : out std_logic;
      ap_idle   : out std_logic;
      ap_ready  : out std_logic;
      din       : in std_logic_vector (31 downto 0);
      ap_return : out std_logic_vector (31 downto 0)
    );
  end component;

  signal idle            : std_logic;
  signal component_ready : std_logic;

begin

  my_trunc_U1 : component my_trunc
    port map(
      ap_clk    => clk,
      ap_rst    => rst,
      ap_start  => pValid,
      ap_done   => valid,
      ap_idle   => idle,
      ap_ready  => component_ready,
      din       => dataInArray,
      ap_return => dataOutArray
    );

    ready <= idle and nReady;

  end architecture;
