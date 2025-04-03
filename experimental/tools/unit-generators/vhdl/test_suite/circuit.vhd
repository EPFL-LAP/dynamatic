library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity circuit is
  port (
    clk, rst : in std_logic;

    speculator_ins : in std_logic_vector(1 - 1 downto 0);
    speculator_ins_valid : in std_logic;
    speculator_ins_spec : in std_logic_vector(0 downto 0);
    speculator_ins_ready : out std_logic;
    speculator_trigger_valid : in std_logic;
    speculator_trigger_spec : in std_logic_vector(0 downto 0);
    speculator_trigger_ready : out std_logic;
    speculator_outs : out std_logic_vector(1 - 1 downto 0);
    speculator_outs_valid : out std_logic;
    speculator_outs_spec : out std_logic_vector(0 downto 0);
    speculator_outs_ready : in std_logic;
    speculator_ctrl_save : out std_logic_vector(0 downto 0);
    speculator_ctrl_save_valid : out std_logic;
    speculator_ctrl_save_ready : in std_logic;
    speculator_ctrl_commit : out std_logic_vector(0 downto 0);
    speculator_ctrl_commit_valid : out std_logic;
    speculator_ctrl_commit_ready : in std_logic;
    speculator_ctrl_sc_branch : out std_logic_vector(0 downto 0);
    speculator_ctrl_sc_branch_valid : out std_logic;
    speculator_ctrl_sc_branch_ready : in std_logic;

    sc_ins : in std_logic_vector(32 - 1 downto 0);
    sc_ins_valid : in std_logic;
    sc_ins_spec : in std_logic_vector(0 downto 0);
    sc_ins_ready : out std_logic;
    sc_outs : out std_logic_vector(32 - 1 downto 0);
    sc_outs_valid : out std_logic;
    sc_outs_spec : out std_logic_vector(0 downto 0);
    sc_outs_ready : in std_logic
  );
end entity;

architecture arch of circuit is
  signal merge_ins : data_array(2 - 1 downto 0)(3 - 1 downto 0);
  signal merge_ins_valid : std_logic_vector(2 - 1 downto 0);
  signal merge_ins_ready : std_logic_vector(2 - 1 downto 0);
  signal merge_outs : std_logic_vector(3 - 1 downto 0);
  signal merge_outs_valid : std_logic;
  signal merge_outs_ready : std_logic;
  signal speculator_ctrl_sc_save : std_logic_vector(2 downto 0);
  signal speculator_ctrl_sc_save_valid : std_logic;
  signal speculator_ctrl_sc_save_ready : std_logic;
  signal speculator_ctrl_sc_commit : std_logic_vector(2 downto 0);
  signal speculator_ctrl_sc_commit_valid : std_logic;
  signal speculator_ctrl_sc_commit_ready : std_logic;
begin
  speculator : entity work.speculator(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => speculator_ins,
      ins_valid => speculator_ins_valid,
      ins_spec => speculator_ins_spec,
      ins_ready => speculator_ins_ready,
      trigger_valid => speculator_trigger_valid,
      trigger_spec => speculator_trigger_spec,
      trigger_ready => speculator_trigger_ready,
      outs => speculator_outs,
      outs_valid => speculator_outs_valid,
      outs_spec => speculator_outs_spec,
      outs_ready => speculator_outs_ready,
      ctrl_save => speculator_ctrl_save,
      ctrl_save_valid => speculator_ctrl_save_valid,
      ctrl_save_ready => speculator_ctrl_save_ready,
      ctrl_commit => speculator_ctrl_commit,
      ctrl_commit_valid => speculator_ctrl_commit_valid,
      ctrl_commit_ready => speculator_ctrl_commit_ready,
      ctrl_sc_save => speculator_ctrl_sc_save,
      ctrl_sc_save_valid => speculator_ctrl_sc_save_valid,
      ctrl_sc_save_ready => speculator_ctrl_sc_save_ready,
      ctrl_sc_commit => speculator_ctrl_sc_commit,
      ctrl_sc_commit_valid => speculator_ctrl_sc_commit_valid,
      ctrl_sc_commit_ready => speculator_ctrl_sc_commit_ready,
      ctrl_sc_branch => speculator_ctrl_sc_branch,
      ctrl_sc_branch_valid => speculator_ctrl_sc_branch_valid,
      ctrl_sc_branch_ready => speculator_ctrl_sc_branch_ready
    );

  merge_ins(0) <= speculator_ctrl_sc_save;
  merge_ins(1) <= speculator_ctrl_sc_commit;
  merge_ins_valid(0) <= speculator_ctrl_sc_save_valid;
  merge_ins_valid(1) <= speculator_ctrl_sc_commit_valid;
  speculator_ctrl_sc_save_ready <= merge_ins_ready(0);
  speculator_ctrl_sc_commit_ready <= merge_ins_ready(1);
  merge : entity work.merge(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => merge_ins,
      ins_valid => merge_ins_valid,
      ins_ready => merge_ins_ready,
      outs => merge_outs,
      outs_valid => merge_outs_valid,
      outs_ready => merge_outs_ready
    );

  sc : entity work.spec_save_commit(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => sc_ins,
      ins_valid => sc_ins_valid,
      ins_spec => sc_ins_spec,
      ins_ready => sc_ins_Ready,
      ctrl => merge_outs,
      ctrl_valid => merge_outs_valid,
      ctrl_ready => merge_outs_ready,
      outs => sc_outs,
      outs_valid => sc_outs_valid,
      outs_spec => sc_outs_spec,
      outs_ready => sc_outs_ready
    );
end architecture;
