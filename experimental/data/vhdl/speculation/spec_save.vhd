library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_save_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic; -- not used
    ctrl      : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    outs_ready : in  std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end spec_save_with_tag;

architecture arch of spec_save_with_tag is

signal fork_outs : data_array(1 downto 0)(DATA_TYPE - 1 downto 0);
signal fork_outs_valid : std_logic_vector(1 downto 0);
signal fork_outs_ready : std_logic_vector(1 downto 0);

signal buff_outs : std_logic_vector(DATA_TYPE - 1 downto 0);
signal buff_outs_valid : std_logic;
signal buff_outs_ready : std_logic;

signal branch_resend_trueOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_resend_trueOut_valid : std_logic;
signal branch_resend_trueOut_ready : std_logic;
signal branch_resend_falseOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_resend_falseOut_valid : std_logic;
signal branch_resend_falseOut_ready : std_logic;

signal merge_ins : data_array(1 downto 0)(DATA_TYPE - 1 downto 0);
signal merge_ins_valid : std_logic_vector(1 downto 0);
signal merge_ins_ready : std_logic_vector(1 downto 0);

begin

fork_in: entity work.handshake_fork(arch)
  generic map(
    SIZE => 2,
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    ins => ins,
    ins_valid => ins_valid,
    ins_ready => ins_ready,
    outs => fork_outs,
    outs_valid => fork_outs_valid,
    outs_ready => fork_outs_ready
  );

buff: entity work.tfifo(arch)
  generic map (
    NUM_SLOTS => 1,
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    ins => fork_outs(1),
    ins_valid => fork_outs_valid(1),
    ins_ready => fork_outs_ready(1),
    outs => buff_outs,
    outs_valid => buff_outs_valid,
    outs_ready => buff_outs_ready
  );

branch_resend_trueOut_ready <= '1'; -- sink
branch_resend: entity work.cond_br(arch)
  generic map(
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    data => buff_outs,
    data_valid => buff_outs_valid,
    data_ready => buff_outs_ready,
    condition => ctrl,
    condition_valid => ctrl_valid,
    condition_ready => ctrl_ready,
    trueOut => branch_resend_trueOut,
    trueOut_valid => branch_resend_trueOut_valid,
    trueOut_ready => branch_resend_trueOut_ready,
    falseOut => branch_resend_falseOut,
    falseOut_valid => branch_resend_falseOut_valid,
    falseOut_ready => branch_resend_falseOut_ready
  );

merge_ins <= (branch_resend_falseOut, fork_outs(0));
merge_ins_valid <= (branch_resend_falseOut_valid, fork_outs_valid(0));
branch_resend_falseOut_ready <= merge_ins_ready(1);
fork_outs_ready(0) <= merge_ins_ready(0);
merge_out: entity work.merge(arch)
  generic map (
    SIZE => 2,
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    ins => merge_ins,
    ins_valid => merge_ins_valid,
    ins_ready => merge_ins_ready,
    outs => outs,
    outs_valid => outs_valid,
    outs_ready => outs_ready
  );

  outs_spec_tag <= '0';
end arch;
