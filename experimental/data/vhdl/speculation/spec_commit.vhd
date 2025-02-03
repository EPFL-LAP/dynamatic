library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_commit_with_tag is
  generic (
    DATA_TYPE : integer  -- use normal data size, eg- 32
  );
  port (
    clk, rst : in  std_logic;
    -- inputs
    ins : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec_tag : in std_logic;
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    outs_ready : in std_logic;
    -- outputs
    outs : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end spec_commit_with_tag;

architecture arch of spec_commit_with_tag is

signal fifo_disc_outs : std_logic_vector(0 downto 0);
signal fifo_disc_outs_valid : std_logic;
signal fifo_disc_outs_ready : std_logic;

signal branch_in_condition : std_logic_vector(0 downto 0);
signal branch_in_condition_ready : std_logic;
signal branch_in_trueOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_in_trueOut_valid : std_logic;
signal branch_in_trueOut_ready : std_logic;
signal branch_in_falseOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_in_falseOut_valid : std_logic;
signal branch_in_falseOut_ready : std_logic;

signal buff_outs : std_logic_vector(DATA_TYPE - 1 downto 0);
signal buff_outs_valid : std_logic;
signal buff_outs_ready : std_logic;

signal branch_disc_trueOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_disc_trueOut_valid : std_logic;
signal branch_disc_trueOut_ready : std_logic;
signal branch_disc_falseOut : std_logic_vector(DATA_TYPE - 1 downto 0);
signal branch_disc_falseOut_valid : std_logic;
signal branch_disc_falseOut_ready : std_logic;

signal merge_ins : data_array(1 downto 0)(DATA_TYPE - 1 downto 0);
signal merge_ins_valid : std_logic_vector(1 downto 0);
signal merge_ins_ready : std_logic_vector(1 downto 0);

begin

-- Design taken directly from the Speculation 2019 paper

fifo_disc: entity work.tfifo(arch)
  generic map (
    NUM_SLOTS => 1, -- todo
    DATA_TYPE => 1
  )
  port map (
    clk => clk,
    rst => rst,
    ins => ctrl,
    ins_valid => ctrl_valid,
    ins_ready => ctrl_ready,
    outs => fifo_disc_outs,
    outs_valid => fifo_disc_outs_valid,
    outs_ready => fifo_disc_outs_ready
  );

branch_in_condition(0) <= ins_spec_tag;
branch_in: entity work.cond_br(arch)
  generic map (
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    data => ins,
    data_valid => ins_valid,
    data_ready => ins_ready,
    condition => branch_in_condition,
    condition_valid => '1', -- always valid
    condition_ready => branch_in_condition_ready,
    trueOut => branch_in_trueOut,
    trueOut_valid => branch_in_trueOut_valid,
    trueOut_ready => branch_in_trueOut_ready,
    falseOut => branch_in_falseOut,
    falseOut_valid => branch_in_falseOut_valid,
    falseOut_ready => branch_in_falseOut_ready
  );

buff: entity work.tfifo(arch)
  generic map (
    NUM_SLOTS => 1, -- todo
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    ins => branch_in_trueOut,
    ins_valid => branch_in_trueOut_valid,
    ins_ready => branch_in_trueOut_ready,
    outs => buff_outs,
    outs_valid => buff_outs_valid,
    outs_ready => buff_outs_ready
  );

branch_disc_trueOut_ready <= '1'; -- sink
branch_disc: entity work.cond_br(arch)
  generic map (
    DATA_TYPE => DATA_TYPE
  )
  port map (
    clk => clk,
    rst => rst,
    data => buff_outs,
    data_valid => buff_outs_valid,
    data_ready => buff_outs_ready,
    condition => fifo_disc_outs,
    condition_valid => fifo_disc_outs_valid,
    condition_ready => fifo_disc_outs_ready,
    trueOut => branch_disc_trueOut,
    trueOut_valid => branch_disc_trueOut_valid,
    trueOut_ready => branch_disc_trueOut_ready,
    falseOut => branch_disc_falseOut,
    falseOut_valid => branch_disc_falseOut_valid,
    falseOut_ready => branch_disc_falseOut_ready
  );

merge_ins <= (branch_disc_falseOut, branch_in_falseOut);
merge_ins_valid <= (branch_disc_falseOut_valid, branch_in_falseOut_valid);
branch_disc_falseOut_ready <= merge_ins_ready(1);
branch_in_falseOut_ready <= merge_ins_ready(0);

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