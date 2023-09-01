library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity cond_br is generic (BITWIDTH : integer);
port (
  -- inputs
  clk                : in std_logic;
  rst                : in std_logic;
  condition          : in std_logic;
  condition_valid    : in std_logic;
  data               : in std_logic_vector(BITWIDTH - 1 downto 0);
  data_valid         : in std_logic;
  true_result_ready  : in std_logic;
  false_result_ready : in std_logic;
  -- outputs
  condition_ready    : out std_logic;
  data_ready         : out std_logic;
  true_result        : out std_logic_vector(BITWIDTH - 1 downto 0);
  true_result_valid  : out std_logic;
  false_result       : out std_logic_vector(BITWIDTH - 1 downto 0);
  false_result_valid : out std_logic);

end cond_br;
architecture arch of cond_br is
  signal joinValid, brReady : std_logic;
begin

  j : entity work.join(arch) generic map(2)
    port map(
    (data_valid, condition_valid),
      brReady,
      joinValid,
      (data_ready, condition_ready));

  cond_br : entity work.branchSimple(arch)
    port map(
      condition,
      joinValid,
      true_result_ready,
      false_result_ready,
      true_result_valid,
      false_result_valid,
      brReady);

  process (data)
  begin
    true_result  <= data;
    false_result <= data;
  end process;

end architecture;
