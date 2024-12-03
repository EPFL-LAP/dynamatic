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

  signal condition_inner : data_array(0 downto 0)(0 downto 0);
  signal dataInArray : data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specInArray : data_array(1 downto 0)(0 downto 0);
  signal pValidArray : std_logic_vector(1 downto 0);
  signal readyArray : std_logic_vector(1 downto 0);

  signal dataOutArray : data_array(1 downto 0)(DATA_TYPE - 1 downto 0);
  signal specOutArray : data_array(1 downto 0)(DATA_TYPE - 1 downto 0);
  signal validArray : std_logic_vector(1 downto 0);
  signal nReadyArray : std_logic_vector(1 downto 0);
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

  condition_inner(0)(0) <= '0';
  dataInArray(0) <= data_inner;
  specInArray(0)(0) <= data_spec_tag_inner;
  specInArray(1)(0) <= spec_tag_data_spec_tag_inner;
  pValidArray <= spec_tag_data_valid_inner & data_valid_inner;
  spec_tag_data_ready_inner <= readyArray(1);
  data_ready_inner <= readyArray(0);
  trueOut <= dataOutArray(0);
  falseOut <= dataOutArray(1);
  trueOut_spec_tag <= specOutArray(0)(0);
  falseOut_spec_tag <= specOutArray(1)(0);
  trueOut_valid <= validArray(0);
  falseOut_valid <= validArray(1);
  nReadyArray(0) <= trueOut_ready;
  nReadyArray(1) <= falseOut_ready;
  speculating_branch : entity work.speculating_branch(arch)
    generic map(
      INPUTS => 2,
      OUTPUTS => 2,
      DATA_SIZE_IN => DATA_TYPE,
      DATA_SIZE_OUT => DATA_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      condition => condition_inner,
      dataInArray => dataInArray,
      specInArray => specInArray,
      pValidArray => pValidArray,
      readyArray => readyArray,
      dataOutArray => dataOutArray,
      specOutArray => specOutArray,
      validArray => validArray,
      nReadyArray => nReadyArray
    );
end architecture;
