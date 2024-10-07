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
    enable: in std_logic;
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
begin
  dataInArray(0) <= ins;
  specInArray(0)(0) <= ins_spec_tag;
  enableInArray(0)(0) <= enable;
  pValidArray <= enable_valid & ins_valid;
  enable_ready <= readyArray(1);
  ins_ready <= readyArray(0);
  outs <= dataOutArray(0);
  outs_spec_tag <= specOutArray(0)(0);
  ctrl_save <= saveOutArray(0);
  ctrl_commit <= commitOutArray(0);
  ctrl_sc_save <= scOut0Array(0);
  ctrl_sc_commit <= scOut1Array(0);
  ctrl_sc_branch <= scBranchOutArray(0);
  nReadyArray(5) <= ctrl_save_ready;
  nReadyArray(4) <= ctrl_commit_ready;
  nReadyArray(3) <= ctrl_sc_commit_ready;
  nReadyArray(2) <= ctrl_sc_save_ready;
  nReadyArray(1) <= ctrl_sc_branch_ready;
  nReadyArray(0) <= outs_ready;
  ctrl_save_valid <= validArray(5);
  ctrl_commit_valid <= validArray(4);
  ctrl_sc_commit_valid <= validArray(3);
  ctrl_sc_save_valid <= validArray(2);
  ctrl_sc_branch_valid <= validArray(1);
  outs_valid <= validArray(0);
  -- temp
  ctrl_save_spec_tag <= '0';
  ctrl_commit_spec_tag <= '0';
  ctrl_sc_save_spec_tag <= '0';
  ctrl_sc_commit_spec_tag <= '0';
  ctrl_sc_branch_spec_tag <= '0';
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
end architecture;
