library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity select is
  generic (
    BITWIDTH : integer
  );
  -- llvm select: operand(0) is condition, operand(1) is true, operand(2) is false
  -- here, dataInArray(0) is true, dataInArray(1) is false operand
  port (
    clk, rst : in std_logic;
    --dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValidArray  : in std_logic_vector(2 downto 0);
    nReady       : in std_logic;
    valid        : out std_logic;
    readyArray   : out std_logic_vector(2 downto 0);
    condition    : in std_logic;
  );

end select;

architecture arch of select is
  signal ee, validInternal : std_logic;
  signal kill0, kill1      : std_logic;
  signal antitokenStop     : std_logic;
  signal g0, g1            : std_logic;
begin

  ee            <= pValidArray(0) and ((not condition and pValidArray(2)) or (condition and pValidArray(1))); --condition and one input
  validInternal <= ee and not antitokenStop;                                                                  -- propagate ee if not stopped by antitoken

  g0 <= not pValidArray(1) and validInternal and nReady;
  g1 <= not pValidArray(2) and validInternal and nReady;

  valid         <= validInternal;
  readyArray(1) <= (not pValidArray(1)) or (validInternal and nReady) or kill0; -- normal join or antitoken
  readyArray(2) <= (not pValidArray(2)) or (validInternal and nReady) or kill1; --normal join or antitoken
  readyArray(0) <= (not pValidArray(0)) or (validInternal and nReady);          --like normal join

  dataOutArray <= inShiftBy when (condition = '0') else
    inToShift;

  Antitokens : entity work.antitokens
    port map(
      clk, rst,
      pValidArray(2), pValidArray(1),
      kill1, kill0,
      g1, g0,
      antitokenStop);

end architecture;
