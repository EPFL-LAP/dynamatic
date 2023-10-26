library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity elasticBufferDummy is
  generic (
    SIZE          : integer;
    INPUTS        : integer := 32;
    DATA_SIZE_IN  : integer;
    DATA_SIZE_OUT : integer
  );
  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    dataOutArray : out std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
    ReadyArray   : out std_logic_vector(0 downto 0);
    ValidArray   : out std_logic_vector(0 downto 0);
    nReadyArray  : in std_logic_vector(0 downto 0);
    pValidArray  : in std_logic_vector(0 downto 0));
end elasticBufferDummy;

architecture arch of elasticBufferDummy is

begin

  dataOutArray  <= dataInArray;
  ValidArray(0) <= pValidArray(0);
  ReadyArray(0) <= nReadyArray(0);

end arch;
