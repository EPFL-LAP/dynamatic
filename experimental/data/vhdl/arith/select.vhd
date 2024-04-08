library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity select_node is
  generic (
    BITWIDTH : integer
  );

  port (
    -- inputs
    clk               : in std_logic;
    rst               : in std_logic;
    condition         : in std_logic;
    condition_valid   : in std_logic;
    true_value        : in std_logic_vector(BITWIDTH - 1 downto 0);
    true_value_valid  : in std_logic;
    false_value       : in std_logic_vector(BITWIDTH - 1 downto 0);
    false_value_valid : in std_logic;
    result_ready      : in std_logic;
    -- outputs
    condition_ready   : out std_logic;
    true_value_ready  : out std_logic;
    false_value_ready : out std_logic;
    result            : out std_logic_vector(BITWIDTH - 1 downto 0);
    result_valid      : out std_logic);

end entity;

architecture arch of select_node is
  signal ee, validInternal : std_logic;
  signal kill0, kill1      : std_logic;
  signal antitokenStop     : std_logic;
  signal g0, g1            : std_logic;
begin

  ee            <= condition_valid and ((not condition and false_value_valid) or (condition and true_value_valid)); --condition and one input
  validInternal <= ee and not antitokenStop;                                                                        -- propagate ee if not stopped by antitoken

  g0 <= not true_value_valid and validInternal and result_ready;
  g1 <= not false_value_valid and validInternal and result_ready;

  result_valid      <= validInternal;
  true_value_ready  <= (not true_value_valid) or (validInternal and result_ready) or kill0;  -- normal join or antitoken
  false_value_ready <= (not false_value_valid) or (validInternal and result_ready) or kill1; --normal join or antitoken
  condition_ready   <= (not condition_valid) or (validInternal and result_ready);            --like normal join

  result <= false_value when (condition = '0') else
    true_value;

  Antitokens : entity work.antitokens
    port map(
      clk, rst,
      false_value_valid, true_value_valid,
      kill1, kill0,
      g1, g0,
      antitokenStop);

end architecture;
