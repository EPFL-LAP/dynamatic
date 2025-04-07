from generators.handshake.tfifo import generate_tfifo
from generators.handshake.cond_br import generate_cond_br
from generators.handshake.merge import generate_merge
from generators.support.signal_manager import generate_signal_manager, get_concat_extra_signals_bitwidth
from generators.support.utils import data


def generate_spec_commit(name, params):
  bitwidth = params["bitwidth"]
  extra_signals = params["extra_signals"]

  # Always contains spec signal
  if len(extra_signals) > 1:
    return _generate_spec_commit_signal_manager(name,  bitwidth, extra_signals)
  return _generate_spec_commit(name, bitwidth)


def _generate_spec_commit(name, bitwidth):
  fifo_disc_name = f"{name}_fifo_disc"
  cond_br_name = f"{name}_cond_br"
  buff_name = f"{name}_buff"
  merge_name = f"{name}_merge"

  dependencies = \
      generate_tfifo(fifo_disc_name, {
          "num_slots": 1,
          "bitwidth": 1
      }) + \
      generate_cond_br(cond_br_name, {
          "bitwidth": bitwidth,
      }) + \
      generate_tfifo(buff_name, {
          "num_slots": 1,
          "bitwidth": bitwidth
      }) + \
      generate_merge(merge_name, {
          "size": 2,
          "bitwidth": bitwidth,
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of spec_commit
entity {name} is
  port (
    clk, rst : in  std_logic;
    {data(f"ins : in std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    {data(f"outs : out std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of spec_commit
architecture arch of {name} is
  signal fifo_disc_outs : std_logic_vector(0 downto 0);
  signal fifo_disc_outs_valid : std_logic;
  signal fifo_disc_outs_ready : std_logic;

  signal branch_in_condition : std_logic_vector(0 downto 0);
  signal branch_in_condition_ready : std_logic;
  {data(f"signal branch_in_trueOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_in_trueOut_valid : std_logic;
  signal branch_in_trueOut_ready : std_logic;
  {data(f"signal branch_in_falseOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_in_falseOut_valid : std_logic;
  signal branch_in_falseOut_ready : std_logic;

  {data(f"signal buff_outs : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal buff_outs_valid : std_logic;
  signal buff_outs_ready : std_logic;

  {data(f"signal branch_disc_trueOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_disc_trueOut_valid : std_logic;
  signal branch_disc_trueOut_ready : std_logic;
  {data(f"signal branch_disc_falseOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_disc_falseOut_valid : std_logic;
  signal branch_disc_falseOut_ready : std_logic;

  {data(f"signal merge_ins : data_array(1 downto 0)({bitwidth} - 1 downto 0);", bitwidth)}
  signal merge_ins_valid : std_logic_vector(1 downto 0);
  signal merge_ins_ready : std_logic_vector(1 downto 0);
begin
  -- Design taken directly from the Speculation 2019 paper
  fifo_disc: entity work.{fifo_disc_name}(arch)
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

  branch_in_condition <= ins_spec;
  branch_in: entity work.{cond_br_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      {data("data => ins,", bitwidth)}
      data_valid => ins_valid,
      data_ready => ins_ready,
      condition => branch_in_condition,
      condition_valid => '1', -- always valid
      condition_ready => branch_in_condition_ready,
      {data("trueOut => branch_in_trueOut,", bitwidth)}
      trueOut_valid => branch_in_trueOut_valid,
      trueOut_ready => branch_in_trueOut_ready,
      {data("falseOut => branch_in_falseOut,", bitwidth)}
      falseOut_valid => branch_in_falseOut_valid,
      falseOut_ready => branch_in_falseOut_ready
    );

  buff: entity work.{buff_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      {data("ins => branch_in_trueOut,", bitwidth)}
      ins_valid => branch_in_trueOut_valid,
      ins_ready => branch_in_trueOut_ready,
      {data("outs => buff_outs,", bitwidth)}
      outs_valid => buff_outs_valid,
      outs_ready => buff_outs_ready
    );

  branch_disc_trueOut_ready <= '1'; -- sink
  branch_disc: entity work.{cond_br_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      {data("data => buff_outs,", bitwidth)}
      data_valid => buff_outs_valid,
      data_ready => buff_outs_ready,
      condition => fifo_disc_outs,
      condition_valid => fifo_disc_outs_valid,
      condition_ready => fifo_disc_outs_ready,
      {data("trueOut => branch_disc_trueOut,", bitwidth)}
      trueOut_valid => branch_disc_trueOut_valid,
      trueOut_ready => branch_disc_trueOut_ready,
      {data("falseOut => branch_disc_falseOut,", bitwidth)}
      falseOut_valid => branch_disc_falseOut_valid,
      falseOut_ready => branch_disc_falseOut_ready
    );

  {data("merge_ins <= (branch_disc_falseOut, branch_in_falseOut);", bitwidth)}
  merge_ins_valid <= (branch_disc_falseOut_valid, branch_in_falseOut_valid);
  branch_disc_falseOut_ready <= merge_ins_ready(1);
  branch_in_falseOut_ready <= merge_ins_ready(0);

  merge_out: entity work.{merge_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      {data("ins => merge_ins,", bitwidth)}
      ins_valid => merge_ins_valid,
      ins_ready => merge_ins_ready,
      {data("outs => outs,", bitwidth)}
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  return dependencies + entity + architecture


def _generate_spec_commit_signal_manager(name, bitwidth, extra_signals):
  extra_signals_without_spec = extra_signals.copy()
  extra_signals_without_spec.pop("spec")

  extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
      extra_signals)
  return generate_signal_manager(name, {
      "type": "concat",
      "in_ports": [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }, {
          "name": "ctrl",
          "bitwidth": 1
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals_without_spec,
      }],
      "extra_signals": extra_signals_without_spec,
      "ignore_ports": ["ctrl"]
  }, lambda name: _generate_spec_commit(name, bitwidth + extra_signals_bitwidth - 1))
