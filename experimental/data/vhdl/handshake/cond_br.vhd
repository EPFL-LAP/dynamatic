library ieee;
use ieee.std_logic_1164.all;

entity cond_br is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst        : in std_logic;
    condition       : in std_logic;
    condition_valid : in std_logic;
    data            : in std_logic_vector(BITWIDTH - 1 downto 0);
    data_valid      : in std_logic;
    trueOut_ready   : in std_logic;
    falseOut_ready  : in std_logic;
    -- outputs
    trueOut         : out std_logic_vector(BITWIDTH - 1 downto 0);
    trueOut_valid   : out std_logic;
    falseOut        : out std_logic_vector(BITWIDTH - 1 downto 0);
    falseOut_valid  : out std_logic;
    condition_ready : out std_logic;
    data_ready      : out std_logic
  );

end entity;
architecture arch of cond_br is
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
      condition,
      joinValid,
      (trueOut_ready,
      falseOut_ready),
      out2_array,
      brReady);

  process (data)
  begin
    trueOut  <= data;
    falseOut <= data;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;

entity branch_simple is
  port (
    -- inputs
    condition  : in std_logic;
    valid      : in std_logic;
    outs_ready : in std_logic_vector(1 downto 0);
    -- outputs
    ins_valid : out std_logic_vector(1 downto 0);
    ins_ready : out std_logic
  );
end branch_simple;

architecture arch of branch_simple is
begin
  ins_valid(1) <= (not condition) and valid;
  ins_valid(0) <= condition and valid;
  ins_ready    <= (outs_ready(1) and not condition) or (outs_ready(0) and condition);
end arch;
