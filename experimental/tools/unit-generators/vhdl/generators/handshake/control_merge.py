from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake
from generators.support.signal_manager.utils.types import ExtraSignals
from generators.handshake.buffers.one_slot_break_r import generate_one_slot_break_r
from generators.handshake.fork import generate_fork
# TODO: Update the normal merge to merge_notehb
from generators.handshake.merge_notehb import generate_merge_notehb
from generators.support.utils import data


def generate_control_merge(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    # e.g., {"tag0": 8, "spec": 1}
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals)
    else:
        return _generate_control_merge(name, size, index_bitwidth, data_bitwidth)


def _generate_control_merge(name, size, index_bitwidth, data_bitwidth):
    merge_name = f"{name}_merge"
    one_slot_break_r_name = f"{name}_one_slot_break_r"
    fork_name = f"{name}_fork"

    dependencies = generate_merge_notehb(merge_name, {"size": size, "bitwidth": data_bitwidth}) + \
        generate_one_slot_break_r(one_slot_break_r_name, {"bitwidth": index_bitwidth + data_bitwidth}) + \
        generate_fork(fork_name, {"size": 2, "bitwidth": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of control_merge
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    {data(f"ins       : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);", data_bitwidth)}
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- data output channel
    {data(f"outs       : out std_logic_vector({data_bitwidth} - 1 downto 0);", data_bitwidth)}
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of control_merge
architecture arch of {name} is
  {data(f"signal merge_outs : std_logic_vector({data_bitwidth} - 1 downto 0);", data_bitwidth)}
  signal merge_outs_valid : std_logic;
  signal buf_ins_ready, buf_outs_valid : std_logic;
  signal fork_ins_ready : std_logic;
  signal index_internal : std_logic_vector({index_bitwidth} - 1 downto 0);
begin
  process (ins_valid)
  begin
    index_internal <= ({index_bitwidth} - 1 downto 0 => '0');
    for i in 0 to ({size} - 1) loop
      if (ins_valid(i) = '1') then
        index_internal <= std_logic_vector(to_unsigned(i, {index_bitwidth}));
        exit;
      end if;
    end loop;
  end process;

  merge_ins : entity work.{merge_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {data("ins => ins,", data_bitwidth)}
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      {data("outs => merge_outs,", data_bitwidth)}
      outs_valid => merge_outs_valid,
      outs_ready => buf_ins_ready
    );

  one_slot_break_r : entity work.{one_slot_break_r_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins({index_bitwidth - 1} downto 0) => index_internal,
      {data(f"ins({data_bitwidth + index_bitwidth - 1} downto {index_bitwidth}) => merge_outs,", data_bitwidth)}
      ins_valid => merge_outs_valid,
      ins_ready => buf_ins_ready,
      outs({index_bitwidth - 1} downto 0) => index,
      {data(f"outs({data_bitwidth + index_bitwidth - 1} downto {index_bitwidth}) => outs,", data_bitwidth)}
      outs_valid => buf_outs_valid,
      outs_ready => fork_ins_ready
    );

  fork_valid : entity work.{fork_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => buf_outs_valid,
      ins_ready => fork_ins_ready,
      outs_valid(0) => outs_valid,
      outs_valid(1) => index_valid,
      outs_ready(0) => outs_ready,
      outs_ready(1) => index_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


# TODO: Update CMerge's type constraints and remove this function
def _generate_index_extra_signal_assignments(index_name: str, index_extra_signals: ExtraSignals) -> str:
    """
    Generate VHDL assignments for extra signals on the index port (cmerge).

    Example:
      - index_tag0 <= "0";
    """

    # TODO: Extra signals on the index port are not tested
    index_extra_signals_list = []
    for signal_name in index_extra_signals:
        index_extra_signals_list.append(
            f"  {index_name}_{signal_name} <= {get_default_extra_signal_value(signal_name)};")
    return "\n  ".join(index_extra_signals_list)


def _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals):
    # Generate signal manager entity
    entity = generate_entity(
        name,
        [{
            "name": "ins",
            "bitwidth": data_bitwidth,
            "size": size,
            "extra_signals": extra_signals
        }],
        [{
            "name": "index",
            "bitwidth": index_bitwidth,
            # TODO: Extra signals for index port are not tested
            "extra_signals": extra_signals
        }, {
            "name": "outs",
            "bitwidth": data_bitwidth,
            "extra_signals": extra_signals
        }])

    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_control_merge(
        inner_name, size, index_bitwidth, extra_signals_bitwidth + data_bitwidth)

    assignments = []

    # Concatenate ins data and extra signals to create ins_inner
    assignments.extend(generate_concat_and_handshake(
        "ins", data_bitwidth, "ins_inner", concat_layout, size))

    # Slice outs_inner data to create outs data and extra signals
    assignments.extend(generate_slice_and_handshake(
        "outs_inner", "outs", data_bitwidth, concat_layout))

    # Assign index extra signals (TODO: Remove this)
    index_extra_signal_assignments = _generate_index_extra_signal_assignments(
        "index", extra_signals)

    architecture = f"""
-- Architecture of signal manager (cmerge)
architecture arch of {name} is
  signal ins_inner : data_array({size} - 1 downto 0)({data_bitwidth} + {extra_signals_bitwidth} - 1 downto 0);
  signal ins_inner_valid, ins_inner_ready : std_logic_vector({size} - 1 downto 0);
  signal outs_inner : std_logic_vector({data_bitwidth} + {extra_signals_bitwidth} - 1 downto 0);
  signal outs_inner_valid, outs_inner_ready : std_logic;
begin
  -- Concat/slice data and extra signals
  {"\n  ".join(assignments)}

  -- Assign index extra signals
  {index_extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_inner_valid,
      ins_ready => ins_inner_ready,
      outs => outs_inner,
      outs_valid => outs_inner_valid,
      outs_ready => outs_inner_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready
    );
end architecture;
"""

    return inner + entity + architecture
