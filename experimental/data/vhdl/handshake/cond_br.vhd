library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity cond_br_node is generic (BITWIDTH : integer);
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

end entity;
architecture arch of cond_br_node is
  signal joinValid, brReady : std_logic;
  signal out_array          : std_logic_vector(1 downto 0);
  signal out2_array         : std_logic_vector(1 downto 0);

begin
  out_array(0)  <= data_ready;
  out_array(1)  <= condition_ready;
  out2_array(0) <= true_result_valid;
  out2_array(1) <= false_result_valid;

  j : entity work.join(arch) generic map(2)
    port map(
    (data_valid, condition_valid),
      brReady,
      joinValid,
      out_array);

  cond_brp : entity work.branchSimple(arch)
    port map(
      condition,
      joinValid,
      (true_result_ready,
      false_result_ready),
      out2_array,
      brReady);

  process (data)
  begin
    true_result  <= data;
    false_result <= data;
  end process;

end architecture;
