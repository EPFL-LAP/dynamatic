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
  signal branchInputs_valid, branch_ready : std_logic;
begin
  join : entity work.join(arch)
    generic map(
      SIZE => 2
    )
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cond_br_dataless_with_tag is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_spec_tag : in std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_spec_tag : in std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_spec_tag : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_spec_tag : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

architecture arch of cond_br_dataless_with_tag is
  signal spec_tag : std_logic;
  signal condition_inner : std_logic_vector(0 downto 0);
  signal condition_valid_inner : std_logic;
  signal condition_spec_tag_inner : std_logic;
  signal condition_ready_inner : std_logic;
begin
  spec_tag <= data_spec_tag or condition_spec_tag_inner;
  trueOut_spec_tag <= spec_tag;
  falseOut_spec_tag <= spec_tag;
  tehb0 : entity work.tehb_for_cond_br(arch)
    generic map(
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => condition,
      ins_valid => condition_valid,
      ins_spec_tag => condition_spec_tag,
      ins_ready => condition_ready,
      outs => condition_inner,
      outs_valid => condition_valid_inner,
      outs_spec_tag => condition_spec_tag_inner,
      outs_ready => condition_ready_inner
    );

  cond_br_inner : entity work.cond_br_dataless(arch)
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition_inner,
      condition_valid => condition_valid_inner,
      condition_ready => condition_ready_inner,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );
end architecture;
