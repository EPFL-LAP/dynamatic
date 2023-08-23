library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity d_store is generic (
  ADDR_BITWIDTH : integer;
  DATA_BITWIDTH : integer);
port (
  clk, rst    : in std_logic;
  input_addr  : in std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  dataInArray : in std_logic_vector(DATA_BITWIDTH - 1 downto 0);

  --- interface to previous
  pValidArray : in std_logic_vector(1 downto 0);
  readyArray  : out std_logic_vector(1 downto 0);

  ---interface to next
  dataOutArray : out std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  output_addr  : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  nReadyArray  : in std_logic_vector(1 downto 0);
  validArray   : out std_logic_vector(1 downto 0));

end entity;

architecture arch of d_store is
  signal single_ready : std_logic;
  signal join_valid   : std_logic;

begin

  join_write : entity work.join(arch) generic map(2)
    port map(
      pValidArray,    --pValidArray
      nReadyArray(0), --nready
      join_valid,     --valid
      ReadyArray);    --readyarray

  dataOutArray  <= dataInArray; -- data to LSQ
  validArray(0) <= join_valid;
  output_addr   <= input_addr; -- address to LSQ
  validArray(1) <= join_valid;
end architecture;
