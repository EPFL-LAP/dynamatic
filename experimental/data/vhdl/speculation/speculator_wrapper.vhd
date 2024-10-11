library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity two_tehb_with_tag is
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

architecture arch of two_tehb_with_tag is
  signal outs_inner : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal outs_valid_inner : std_logic;
  signal outs_spec_tag_inner : std_logic;
  signal outs_ready_inner : std_logic;
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
      outs => outs,
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      outs_ready => outs_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity speculator_wrapper_with_tag is
  generic (
    DATA_TYPE : integer;
    FIFO_DEPTH : integer
  );
  port (
    clk, rst: std_logic;
    -- inputs
    ins: in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid: in std_logic;
    ins_spec_tag: in std_logic;
    -- enable is dataless (control token)
    enable_valid: in std_logic;
    enable_spec_tag: in std_logic;
    outs_ready: in std_logic;
    ctrl_save_ready: in std_logic;
    ctrl_commit_ready: in std_logic;
    ctrl_sc_save_ready: in std_logic;
    ctrl_sc_commit_ready: in std_logic;
    ctrl_sc_branch_ready: in std_logic;
    -- outputs
    outs: out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid: out std_logic;
    outs_spec_tag: out std_logic;
    ctrl_save: out std_logic_vector(0 downto 0);
    ctrl_save_valid: out std_logic;
    ctrl_save_spec_tag: out std_logic;
    ctrl_commit: out std_logic_vector(0 downto 0);
    ctrl_commit_valid: out std_logic;
    ctrl_commit_spec_tag: out std_logic;
    ctrl_sc_save: out std_logic_vector(2 downto 0);
    ctrl_sc_save_valid: out std_logic;
    ctrl_sc_save_spec_tag: out std_logic;
    ctrl_sc_commit: out std_logic_vector(2 downto 0);
    ctrl_sc_commit_valid: out std_logic;
    ctrl_sc_commit_spec_tag: out std_logic;
    ctrl_sc_branch: out std_logic_vector(0 downto 0);
    ctrl_sc_branch_valid: out std_logic;
    ctrl_sc_branch_spec_tag: out std_logic;
    ins_ready: out std_logic;
    enable_ready: out std_logic
  );
end entity;

architecture arch of speculator_wrapper_with_tag is
  signal dataInArray: data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specInArray: data_array(0 downto 0)(0 downto 0);
  signal enableInArray: data_array(0 downto 0)(0 downto 0);
  signal pValidArray: std_logic_vector(1 downto 0);
  signal readyArray: std_logic_vector(1 downto 0);
  signal dataOutArray: data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
  signal specOutArray: data_array(0 downto 0)(0 downto 0);
  signal saveOutArray: data_array(0 downto 0)(0 downto 0);
  signal commitOutArray: data_array(0 downto 0)(0 downto 0);
  signal scOut0Array: data_array(0 downto 0)(2 downto 0);
  signal scOut1Array: data_array(0 downto 0)(2 downto 0);
  signal scBranchOutArray: data_array(0 downto 0)(0 downto 0);
  signal nReadyArray: std_logic_vector(5 downto 0);
  signal validArray: std_logic_vector(5 downto 0);

  signal outs_inner: std_logic_vector(DATA_TYPE - 1 downto 0);
  signal outs_valid_inner: std_logic;
  signal outs_spec_tag_inner: std_logic;
  signal outs_ready_inner: std_logic;

  signal ctrl_save_inner: std_logic_vector(0 downto 0);
  signal ctrl_save_valid_inner: std_logic;
  signal ctrl_save_spec_tag_inner: std_logic;
  signal ctrl_save_ready_inner: std_logic;
  signal ctrl_commit_inner: std_logic_vector(0 downto 0);
  signal ctrl_commit_valid_inner: std_logic;
  signal ctrl_commit_spec_tag_inner: std_logic;
  signal ctrl_commit_ready_inner: std_logic;
  signal ctrl_sc_save_inner: std_logic_vector(2 downto 0);
  signal ctrl_sc_save_valid_inner: std_logic;
  signal ctrl_sc_save_spec_tag_inner: std_logic;
  signal ctrl_sc_save_ready_inner: std_logic;
  signal ctrl_sc_commit_inner: std_logic_vector(2 downto 0);
  signal ctrl_sc_commit_valid_inner: std_logic;
  signal ctrl_sc_commit_spec_tag_inner: std_logic;
  signal ctrl_sc_commit_ready_inner: std_logic;
  signal ctrl_sc_branch_inner: std_logic_vector(0 downto 0);
  signal ctrl_sc_branch_valid_inner: std_logic;
  signal ctrl_sc_branch_spec_tag_inner: std_logic;
  signal ctrl_sc_branch_ready_inner: std_logic;
begin
  dataInArray(0) <= ins;
  specInArray(0)(0) <= ins_spec_tag;
  enableInArray(0)(0) <= '0'; -- not used
  pValidArray <= enable_valid & ins_valid;
  enable_ready <= readyArray(1);
  ins_ready <= readyArray(0);
  outs_inner <= dataOutArray(0);
  outs_spec_tag_inner <= specOutArray(0)(0);
  ctrl_save_inner <= saveOutArray(0);
  ctrl_commit_inner <= commitOutArray(0);
  ctrl_sc_save_inner <= scOut0Array(0);
  ctrl_sc_commit_inner <= scOut1Array(0);
  ctrl_sc_branch_inner <= scBranchOutArray(0);
  nReadyArray(5) <= ctrl_save_ready_inner;
  nReadyArray(4) <= ctrl_commit_ready_inner;
  nReadyArray(3) <= ctrl_sc_commit_ready_inner;
  nReadyArray(2) <= ctrl_sc_save_ready_inner;
  nReadyArray(1) <= ctrl_sc_branch_ready_inner;
  nReadyArray(0) <= outs_ready_inner;
  ctrl_save_valid_inner <= validArray(5);
  ctrl_commit_valid_inner <= validArray(4);
  ctrl_sc_commit_valid_inner <= validArray(3);
  ctrl_sc_save_valid_inner <= validArray(2);
  ctrl_sc_branch_valid_inner <= validArray(1);
  outs_valid_inner <= validArray(0);
  -- temp
  ctrl_save_spec_tag_inner <= '0';
  ctrl_commit_spec_tag_inner <= '0';
  ctrl_sc_save_spec_tag_inner <= '0';
  ctrl_sc_commit_spec_tag_inner <= '0';
  ctrl_sc_branch_spec_tag_inner <= '0';
  speculator : entity work.speculator(arch)
    generic map(
      DATA_SIZE_IN => DATA_TYPE,
      DATA_SIZE_OUT => DATA_TYPE,
      FIFO_DEPTH => FIFO_DEPTH
    )
    port map(
      clk => clk,
      rst => rst,
      dataInArray => dataInArray,
      specInArray => specInArray,
      enableInArray => enableInArray,
      pValidArray => pValidArray,
      readyArray => readyArray,
      dataOutArray => dataOutArray,
      specOutArray => specOutArray,
      saveOutArray => saveOutArray,
      commitOutArray => commitOutArray,
      scOut0Array => scOut0Array,
      scOut1Array => scOut1Array,
      scBranchOutArray => scBranchOutArray,
      validArray => validArray,
      nReadyArray => nReadyArray
    );
  tehb_outs : entity work.two_tehb_with_tag(arch)
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
      outs => outs,
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      outs_ready => outs_ready
    );
  tehb_ctrl_save : entity work.two_tehb_with_tag(arch)
    generic map(
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl_save_inner,
      ins_valid => ctrl_save_valid_inner,
      ins_spec_tag => ctrl_save_spec_tag_inner,
      ins_ready => ctrl_save_ready_inner,
      outs => ctrl_save,
      outs_valid => ctrl_save_valid,
      outs_spec_tag => ctrl_save_spec_tag,
      outs_ready => ctrl_save_ready
    );
  tehb_ctrl_commit : entity work.two_tehb_with_tag(arch)
    generic map(
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl_commit_inner,
      ins_valid => ctrl_commit_valid_inner,
      ins_spec_tag => ctrl_commit_spec_tag_inner,
      ins_ready => ctrl_commit_ready_inner,
      outs => ctrl_commit,
      outs_valid => ctrl_commit_valid,
      outs_spec_tag => ctrl_commit_spec_tag,
      outs_ready => ctrl_commit_ready
    );
  tehb_ctrl_sc_save : entity work.two_tehb_with_tag(arch)
    generic map(
      DATA_TYPE => 3
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl_sc_save_inner,
      ins_valid => ctrl_sc_save_valid_inner,
      ins_spec_tag => ctrl_sc_save_spec_tag_inner,
      ins_ready => ctrl_sc_save_ready_inner,
      outs => ctrl_sc_save,
      outs_valid => ctrl_sc_save_valid,
      outs_spec_tag => ctrl_sc_save_spec_tag,
      outs_ready => ctrl_sc_save_ready
    );
  tehb_ctrl_sc_commit : entity work.two_tehb_with_tag(arch)
    generic map(
      DATA_TYPE => 3
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl_sc_commit_inner,
      ins_valid => ctrl_sc_commit_valid_inner,
      ins_spec_tag => ctrl_sc_commit_spec_tag_inner,
      ins_ready => ctrl_sc_commit_ready_inner,
      outs => ctrl_sc_commit,
      outs_valid => ctrl_sc_commit_valid,
      outs_spec_tag => ctrl_sc_commit_spec_tag,
      outs_ready => ctrl_sc_commit_ready
    );
  tehb_ctrl_sc_branch : entity work.two_tehb_with_tag(arch)
    generic map(
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ctrl_sc_branch_inner,
      ins_valid => ctrl_sc_branch_valid_inner,
      ins_spec_tag => ctrl_sc_branch_spec_tag_inner,
      ins_ready => ctrl_sc_branch_ready_inner,
      outs => ctrl_sc_branch,
      outs_valid => ctrl_sc_branch_valid,
      outs_spec_tag => ctrl_sc_branch_spec_tag,
      outs_ready => ctrl_sc_branch_ready
    );
end architecture;
