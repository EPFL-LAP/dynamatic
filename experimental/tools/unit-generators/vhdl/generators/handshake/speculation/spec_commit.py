from generators.handshake.cond_br import generate_cond_br
from generators.handshake.merge import generate_merge
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat, generate_slice
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.utils import data


def generate_spec_commit(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params["extra_signals"]

    # Always contains spec signal
    if len(extra_signals) > 1:
        return _generate_spec_commit_signal_manager(name, bitwidth, extra_signals)
    return _generate_spec_commit(name, bitwidth)


def _generate_spec_commit(name, bitwidth):
    cond_br_name = f"{name}_cond_br"
    merge_name = f"{name}_merge"

    dependencies = \
        generate_cond_br(cond_br_name, {
            "bitwidth": bitwidth,
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
  signal branch_in_condition : std_logic_vector(0 downto 0);
  {data(f"signal branch_in_trueOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_in_trueOut_valid : std_logic;
  signal branch_in_trueOut_ready : std_logic;
  {data(f"signal branch_in_falseOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_in_falseOut_valid : std_logic;
  signal branch_in_falseOut_ready : std_logic;

  {data(f"signal branch_disc_falseOut : std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
  signal branch_disc_falseOut_valid : std_logic;
  signal branch_disc_falseOut_ready : std_logic;

  {data(f"signal merge_ins : data_array(1 downto 0)({bitwidth} - 1 downto 0);", bitwidth)}
  signal merge_ins_valid : std_logic_vector(1 downto 0);
  signal merge_ins_ready : std_logic_vector(1 downto 0);
begin

  branch_in_condition <= ins_spec;

  branch_in: entity work.{cond_br_name}(arch)
    port map (
      clk => clk,
      rst => rst,

      {data("data => ins,", bitwidth)}
      data_valid => ins_valid,
      data_ready => ins_ready,

      condition => branch_in_condition,
      -- Handshaking is common with `data`, keep valid high and ignore ready
      condition_valid => '1',
      condition_ready => open,

      {data("trueOut => branch_in_trueOut,", bitwidth)}
      trueOut_valid => branch_in_trueOut_valid,
      trueOut_ready => branch_in_trueOut_ready,

      {data("falseOut => branch_in_falseOut,", bitwidth)}
      falseOut_valid => branch_in_falseOut_valid,
      falseOut_ready => branch_in_falseOut_ready
    );

  branch_disc: entity work.{cond_br_name}(arch)
    port map (
      clk => clk,
      rst => rst,

      {data("data => branch_in_trueOut,", bitwidth)}
      data_valid => branch_in_trueOut_valid,
      data_ready => branch_in_trueOut_ready,

      condition => ctrl,
      condition_valid => ctrl_valid,
      condition_ready => ctrl_ready,

      -- trueOut sinks
      {data("trueOut => open,", bitwidth)}
      trueOut_valid => open,
      trueOut_ready => '1',

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
    # Concat signals except spec

    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    concat_layout = ConcatLayout(extra_signals_without_spec)
    extra_signals_without_spec_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_spec_commit(
        inner_name, bitwidth + extra_signals_without_spec_bitwidth)

    entity = generate_entity(name, [{
        "name": "ins",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "ctrl",
        "bitwidth": 1,
        "extra_signals": {}
    }], [{
        "name": "outs",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals_without_spec
    }])

    assignments = []

    # Concat ins data and extra signals (except spec) to create ins_concat
    assignments.extend(generate_concat(
        "ins", bitwidth, "ins_concat", concat_layout))

    # Slice outs_concat to create outs data and extra signals (except spec)
    assignments.extend(generate_slice(
        "outs_concat", "outs", bitwidth, concat_layout))

    architecture = f"""
-- Architecture of spec_commit signal manager
architecture arch of {name} is
  signal ins_concat, outs_concat : std_logic_vector({bitwidth + extra_signals_without_spec_bitwidth} - 1 downto 0);
begin
  {"\n  ".join(assignments)}
  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_concat,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs_concat,
      outs_valid => outs_valid,
      outs_ready => outs_ready,
      -- Forward spec signal
      ins_spec => ins_spec
    );
end architecture;
"""

    return inner + entity + architecture
