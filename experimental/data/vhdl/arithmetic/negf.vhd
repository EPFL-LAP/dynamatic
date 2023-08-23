library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
entity negf is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValid       : in std_logic;
    nReady       : in std_logic;
    valid        : out std_logic;
    ready        : out std_logic);
end entity;

architecture arch of negf is

  constant msb_mask : std_logic_vector(31 downto 0) := (31 => '1', others => '0');

begin

  dataOutArray <= dataInArray xor msb_mask;
  valid        <= pValid;
  ready        <= nReady;
end architecture;
