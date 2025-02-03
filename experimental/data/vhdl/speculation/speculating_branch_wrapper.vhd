library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity speculating_branch_wrapper_with_tag is
  generic(
    DATA_TYPE : integer;
    SPEC_TAG_DATA_TYPE : integer
  );
  port(
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    data_valid : in  std_logic;
    data_spec_tag : in std_logic;
    data_ready : out std_logic;
    -- spec_tag_data used for condition
    spec_tag_data       : in  std_logic_vector(SPEC_TAG_DATA_TYPE - 1 downto 0);
    spec_tag_data_valid : in  std_logic;
    spec_tag_data_spec_tag : in std_logic;
    spec_tag_data_ready : out std_logic;
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

architecture arch of speculating_branch_wrapper_with_tag is
  signal data_inner : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal data_valid_inner : std_logic;
  signal data_spec_tag_inner : std_logic;
  signal data_ready_inner : std_logic;
  signal spec_tag_data_inner : std_logic_vector(SPEC_TAG_DATA_TYPE - 1 downto 0);
  signal spec_tag_data_valid_inner : std_logic;
  signal spec_tag_data_spec_tag_inner : std_logic;
  signal spec_tag_data_ready_inner : std_logic;
begin

  data_buf : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      NUM_SLOTS => 32
    )
    port map(
      clk => clk,
      rst => rst,
      ins => data,
      ins_valid => data_valid,
      ins_spec_tag => data_spec_tag,
      ins_ready => data_ready,
      outs => data_inner,
      outs_valid => data_valid_inner,
      outs_spec_tag => data_spec_tag_inner,
      outs_ready => data_ready_inner
    );
  spec_tag_buf : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => SPEC_TAG_DATA_TYPE,
      NUM_SLOTS => 32
    )
    port map(
      clk => clk,
      rst => rst,
      ins => spec_tag_data,
      ins_valid => spec_tag_data_valid,
      ins_spec_tag => spec_tag_data_spec_tag,
      ins_ready => spec_tag_data_ready,
      outs => spec_tag_data_inner,
      outs_valid => spec_tag_data_valid_inner,
      outs_spec_tag => spec_tag_data_spec_tag_inner,
      outs_ready => spec_tag_data_ready_inner
    );

  speculating_branch : entity work.speculating_branch(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      SPEC_TAG_DATA_TYPE => SPEC_TAG_DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      data => data_inner,
      data_valid => data_valid_inner,
      data_spec_tag => data_spec_tag_inner,
      data_ready => data_ready_inner,
      spec_tag_data => spec_tag_data_inner,
      spec_tag_data_valid => spec_tag_data_valid_inner,
      spec_tag_data_spec_tag => spec_tag_data_spec_tag_inner,
      spec_tag_data_ready => spec_tag_data_ready_inner,
      trueOut => trueOut,
      trueOut_valid => trueOut_valid,
      trueOut_spec_tag => trueOut_spec_tag,
      trueOut_ready => trueOut_ready,
      falseOut => falseOut,
      falseOut_valid => falseOut_valid,
      falseOut_spec_tag => falseOut_spec_tag,
      falseOut_ready => falseOut_ready
    );
end architecture;
