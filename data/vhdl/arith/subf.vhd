library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subf_single_precision is
  port (
    -- inputs
    clk : in std_logic;
    rst : in std_logic;
    lhs : in std_logic_vector(32 - 1 downto 0);
    lhs_valid : in std_logic;
    rhs : in std_logic_vector(32 - 1 downto 0);
    rhs_valid : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result : out std_logic_vector(32 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready : out std_logic;
    rhs_ready : out std_logic
  );
end entity;

architecture arch of subf_single_precision is
  constant latency : integer := 9;
  signal join_valid : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;

  -- subf is the same as addf, but we flip the sign bit of rhs
  signal rhs_neg : std_logic_vector(32 - 1 downto 0);

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(32 + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(32 + 1 downto 0);
begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready => oehb_ready,
      -- outputs
      outs_valid => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  oehb : entity work.oehb(arch) generic map(1)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready => oehb_ready,
      ins(0) => '0',
      outs => open
    );

  rhs_neg <= not rhs(32 - 1) & rhs(32 - 2 downto 0);

  buff : entity work.delay_buffer(arch) generic map(latency - 1)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  ieee2nfloat_0 : entity work.InputIEEE_32bit(arch)
    port map(
      X => lhs,
      R => ip_lhs
    );

  ieee2nfloat_1 : entity work.InputIEEE_32bit(arch)
    port map(
      X => rhs_neg,
      R => ip_rhs
    );

  nfloat2ieee : entity work.OutputIEEE_32bit(arch)
    port map(
      X => ip_result,
      R => result
    );

  operator : entity work.FloatingPointAdder(arch)
    port map(
      clk => clk,
      ce => oehb_ready,
      X => ip_lhs,
      Y => ip_rhs,
      R => ip_result
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subf_double_precision is
  port (
    -- inputs
    clk : in std_logic;
    rst : in std_logic;
    lhs : in std_logic_vector(64 - 1 downto 0);
    lhs_valid : in std_logic;
    rhs : in std_logic_vector(64 - 1 downto 0);
    rhs_valid : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result : out std_logic_vector(64 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready : out std_logic;
    rhs_ready : out std_logic
  );
end entity;

architecture arch of subf_double_precision is
  constant latency : integer := 12;
  signal join_valid : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;

  -- subf is the same as addf, but we flip the sign bit of rhs
  signal rhs_neg : std_logic_vector(64 - 1 downto 0);

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(64 + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(64 + 1 downto 0);
begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready => oehb_ready,
      -- outputs
      outs_valid => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );
  oehb : entity work.oehb(arch) generic map(1)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready => oehb_ready,
      ins(0) => '0',
      outs => open
    );

  rhs_neg <= not rhs(64 - 1) & rhs(64 - 2 downto 0);

  buff : entity work.delay_buffer(arch) generic map(latency - 1)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  ieee2nfloat_0 : entity work.InputIEEE_64bit(arch)
    port map(
      X => lhs,
      R => ip_lhs
    );

  ieee2nfloat_1 : entity work.InputIEEE_64bit(arch)
    port map(
      X => rhs_neg,
      R => ip_rhs
    );

  nfloat2ieee : entity work.OutputIEEE_64bit(arch)
    port map(
      X => ip_result,
      R => result
    );

  operator : entity work.FPAdd_64bit(arch)
    port map(
      clk => clk,
      ce => oehb_ready,
      X => ip_lhs,
      Y => ip_rhs,
      R => ip_result
    );

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subf is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk : in std_logic;
    rst : in std_logic;
    lhs : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid : in std_logic;
    rhs : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready : out std_logic;
    rhs_ready : out std_logic
  );
end entity;

architecture arch of subf is
begin
  gen_flopoco_ip :
  if DATA_TYPE = 32 generate
    ip : entity work.subf_single_precision(arch)
      port map(
        clk, rst, lhs, lhs_valid,
        rhs, rhs_valid, result_ready,
        result, result_valid, lhs_ready, rhs_ready
      );
  elsif DATA_TYPE = 64 generate
      ip : entity work.subf_double_precision(arch)
        port map(
          clk, rst, lhs, lhs_valid,
          rhs, rhs_valid, result_ready,
          result, result_valid, lhs_ready, rhs_ready
        );
  else generate
      assert false
      report "subf must operate on 32-bit or 64-bit"
        severity failure;
  end generate;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subf_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    lhs_spec_tag : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    rhs_spec_tag : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    result_spec_tag : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of subf_with_tag is
  signal spec_tag_inner : std_logic_vector(0 downto 0);
  signal spec_tag_inner2 : std_logic_vector(0 downto 0);
  signal spec_tag_tfifo_pvalid : std_logic;
  signal spec_tag_tfifo_nready : std_logic;
begin
  spec_tag_inner(0) <= lhs_spec_tag or rhs_spec_tag;
  spec_tag_tfifo_pvalid <= lhs_valid and rhs_valid and lhs_ready and rhs_ready;
  spec_tag_tfifo_nready <= result_valid and result_ready;
  result_spec_tag <= spec_tag_inner2(0);
  spec_tag_tfifo : entity work.tfifo(arch)
    generic map(
      NUM_SLOTS => 16,
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => spec_tag_inner,
      ins_valid => spec_tag_tfifo_pvalid,
      ins_ready => open,
      outs => spec_tag_inner2,
      outs_valid => open,
      outs_ready => spec_tag_tfifo_nready
    );
  subf : entity work.subf(arch)
    generic map(DATA_TYPE)
    port map(
      clk, rst, lhs, lhs_valid,
      rhs, rhs_valid, result_ready,
      result, result_valid, lhs_ready, rhs_ready
    );
end architecture;
