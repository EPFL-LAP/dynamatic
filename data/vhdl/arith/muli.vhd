library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mul_4_stage is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk : in  std_logic;
    ce  : in  std_logic;
    a   : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    b   : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    p   : out std_logic_vector(DATA_TYPE - 1 downto 0));
end entity;

architecture behav of mul_4_stage is

  signal a_reg : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal b_reg : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q0    : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q1    : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q2    : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal mul   : std_logic_vector(DATA_TYPE - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), DATA_TYPE));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity muli is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk, rst     : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of muli is
  constant LATENCY                          : integer := 4;
  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector(DATA_TYPE - 1 downto 0);
begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  multiply_unit : entity work.mul_4_stage(behav) generic map(DATA_TYPE)
    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => lhs,
      b   => rhs,
      p   => result
    );

  buff : entity work.delay_buffer(arch) generic map(LATENCY - 1)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.oehb(arch) generic map (DATA_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity muli_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk, rst     : in std_logic;
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

architecture arch of muli_with_tag is
  signal spec_tag_inner : std_logic_vector(0 downto 0);
  signal spec_tag_inner2 : std_logic_vector(0 downto 0);
begin
  spec_tag_inner(0) <= lhs_spec_tag or rhs_spec_tag;
  result_spec_tag <= spec_tag_inner2(0);
  spec_tag_tfifo : entity work.tfifo(arch)
    generic map(
      NUM_SLOTS => 4, -- 5?
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => spec_tag_inner,
      ins_valid => lhs_valid and rhs_valid and lhs_ready and rhs_ready,
      ins_ready => open,
      outs => spec_tag_inner2,
      outs_valid => open,
      outs_ready => result_valid and result_ready
    );
  muli_inner : entity work.muli(arch) generic map(DATA_TYPE)
    port map(
      clk          => clk,
      rst          => rst,
      lhs          => lhs,
      lhs_valid    => lhs_valid,
      rhs          => rhs,
      rhs_valid    => rhs_valid,
      result_ready => result_ready,
      result       => result,
      result_valid => result_valid,
      lhs_ready    => lhs_ready,
      rhs_ready    => rhs_ready
    );
end architecture;
