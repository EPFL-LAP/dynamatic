library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

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
  -- temp
  ctrl_save_spec_tag_inner <= '0';
  ctrl_commit_spec_tag_inner <= '0';
  ctrl_sc_save_spec_tag_inner <= '0';
  ctrl_sc_commit_spec_tag_inner <= '0';
  ctrl_sc_branch_spec_tag_inner <= '0';
  speculator : entity work.speculator(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      FIFO_DEPTH => FIFO_DEPTH
    )
    port map(
      clk => clk,
      rst => rst,

      ins => ins,
      ins_valid => ins_valid,
      ins_spec_tag => ins_spec_tag,
      ins_ready => ins_ready,

      enable_valid => enable_valid,
      enable_ready => enable_ready,

      outs => outs_inner,
      outs_valid => outs_valid_inner,
      outs_spec_tag => outs_spec_tag_inner,
      outs_ready => outs_ready_inner,

      ctrl_save => ctrl_save_inner,
      ctrl_save_valid => ctrl_save_valid_inner,
      ctrl_save_ready => ctrl_save_ready_inner,

      ctrl_commit => ctrl_commit_inner,
      ctrl_commit_valid => ctrl_commit_valid_inner,
      ctrl_commit_ready => ctrl_commit_ready_inner,

      ctrl_sc_save => ctrl_sc_save_inner,
      ctrl_sc_save_valid => ctrl_sc_save_valid_inner,
      ctrl_sc_save_ready => ctrl_sc_save_ready_inner,

      ctrl_sc_commit => ctrl_sc_commit_inner,
      ctrl_sc_commit_valid => ctrl_sc_commit_valid_inner,
      ctrl_sc_commit_ready => ctrl_sc_commit_ready_inner,

      ctrl_sc_branch => ctrl_sc_branch_inner,
      ctrl_sc_branch_valid => ctrl_sc_branch_valid_inner,
      ctrl_sc_branch_ready => ctrl_sc_branch_ready_inner
    );
  tehb_outs : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => DATA_TYPE,
      NUM_SLOTS => 32
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
  tehb_ctrl_save : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => 1,
      NUM_SLOTS => 32
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
  tehb_ctrl_commit : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => 1,
      NUM_SLOTS => 32
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
  tehb_ctrl_sc_save : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => 3,
      NUM_SLOTS => 32
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
  tehb_ctrl_sc_commit : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => 3,
      NUM_SLOTS => 32
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
  tehb_ctrl_sc_branch : entity work.tfifo_with_tag(arch)
    generic map(
      DATA_TYPE => 1,
      NUM_SLOTS => 32
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
