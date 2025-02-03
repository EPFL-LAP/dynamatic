library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity speculating_branch_with_tag is
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
end speculating_branch_with_tag;

architecture arch of speculating_branch_with_tag is
  signal cond_br_condition : std_logic_vector(0 downto 0);
begin

  cond_br_condition(0) <= spec_tag_data_spec_tag;
  cond_br : entity work.cond_br_with_tag
    generic map (
      DATA_TYPE => DATA_TYPE
    )
    port map (
      clk => clk,
      rst => rst,
      data => data,
      data_valid => data_valid,
      data_spec_tag => data_spec_tag,
      data_ready => data_ready,
      condition => cond_br_condition,
      condition_valid => spec_tag_data_valid,
      condition_spec_tag => data_spec_tag,
      condition_ready => spec_tag_data_ready,
      trueOut => trueOut,
      trueOut_valid => trueOut_valid,
      trueOut_spec_tag => trueOut_spec_tag,
      trueOut_ready => trueOut_ready,
      falseOut => falseOut,
      falseOut_valid => falseOut_valid,
      falseOut_spec_tag => falseOut_spec_tag,
      falseOut_ready => falseOut_ready
    );

end arch;
