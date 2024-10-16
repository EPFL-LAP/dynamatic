library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tehb_for_cond_br is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of tehb_for_cond_br is
  signal outs_inner : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal outs_valid_inner : std_logic;
  signal outs_spec_tag_inner : std_logic;
  signal outs_ready_inner : std_logic;
  signal outs_inner2 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal outs_valid_inner2 : std_logic;
  signal outs_spec_tag_inner2 : std_logic;
  signal outs_ready_inner2 : std_logic;
begin
  tehb0 : entity work.tehb_with_tag(arch)
    generic map(
      DATA_TYPE => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_spec_tag => ins_spec_tag,
      ins_ready => ins_ready,
      outs => outs_inner,
      outs_valid => outs_valid_inner,
      outs_spec_tag => outs_spec_tag_inner,
      outs_ready => outs_ready_inner
    );
  tehb1 : entity work.tehb_with_tag(arch)
    generic map(
      DATA_TYPE => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      ins => outs_inner,
      ins_valid => outs_valid_inner,
      ins_spec_tag => outs_spec_tag_inner,
      ins_ready => outs_ready_inner,
      outs => outs_inner2,
      outs_valid => outs_valid_inner2,
      outs_spec_tag => outs_spec_tag_inner2,
      outs_ready => outs_ready_inner2
    );
  tehb2 : entity work.tehb_with_tag(arch)
    generic map(
      DATA_TYPE => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      ins => outs_inner2,
      ins_valid => outs_valid_inner2,
      ins_spec_tag => outs_spec_tag_inner2,
      ins_ready => outs_ready_inner2,
      outs => outs,
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      outs_ready => outs_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cond_br is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

architecture arch of cond_br is
begin
  control : entity work.cond_br_dataless
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cond_br_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    data_valid : in  std_logic;
    data_spec_tag : in std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_spec_tag : in std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_spec_tag : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_spec_tag : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

architecture arch of cond_br_with_tag is
  signal condition_inner : std_logic_vector(0 downto 0);
  signal condition_valid_inner : std_logic;
  signal condition_spec_tag_inner : std_logic;
  signal condition_ready_inner : std_logic;
begin
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

  control : entity work.cond_br_dataless_with_tag
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_spec_tag   => data_spec_tag,
      data_ready      => data_ready,
      condition       => condition_inner,
      condition_valid => condition_valid_inner,
      condition_spec_tag => condition_spec_tag_inner,
      condition_ready => condition_ready_inner,
      trueOut_valid   => trueOut_valid,
      trueOut_spec_tag => trueOut_spec_tag,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_spec_tag => falseOut_spec_tag,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;
