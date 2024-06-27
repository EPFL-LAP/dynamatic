library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cond_br_dataless is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

architecture arch of cond_br_dataless is
  signal joinValid, brReady : std_logic;
  signal out_array          : std_logic_vector(1 downto 0);
  signal out2_array         : std_logic_vector(1 downto 0);
begin
  data_ready      <= out_array(0);
  condition_ready <= out_array(1);
  trueOut_valid   <= out2_array(0);
  falseOut_valid  <= out2_array(1);

  j : entity work.join(arch) generic map(2)
    port map(
    (data_valid, condition_valid),
      brReady,
      joinValid,
      out_array);

  cond_brp : entity work.branch_simple(arch)
    port map(
      condition(0),
      joinValid,
      (trueOut_ready,
      falseOut_ready),
      out2_array,
      brReady);
end architecture;
