from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake
from generators.support.signal_manager.utils.types import ExtraSignals
from generators.handshake.tehb import generate_tehb
from generators.handshake.merge_notehb import generate_merge_notehb
from generators.handshake.fork import generate_fork


def generate_control_merge(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    # e.g., {"tag0": 8, "spec": 1}
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals)
    elif data_bitwidth == 0:
        return _generate_control_merge_dataless(name, size, index_bitwidth)
    else:
        return _generate_control_merge(name, size, index_bitwidth, data_bitwidth)


def _generate_control_merge_dataless(name, size, index_bitwidth):
    merge_name = f"{name}_merge"
    tehb_name = f"{name}_tehb"
    fork_name = f"{name}_fork"

    dependencies = generate_merge_notehb(merge_name, {"size": size}) + \
        generate_tehb(tehb_name, {"bitwidth": index_bitwidth}) + \
        generate_fork(fork_name, {"size": 2, "bitwidth": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of control_merge_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- data output channel
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
-- Architecture of control_merge_dataless
architecture arch of {name} is
  signal index_tehb                                               : std_logic_vector ({index_bitwidth} - 1 downto 0);
  signal dataAvailable, readyToFork, tehbOut_valid, tehbOut_ready : std_logic;
begin
  process (ins_valid)
  begin
    index_tehb <= ({index_bitwidth} - 1 downto 0 => '0');
    for i in 0 to ({size} - 1) loop
      if (ins_valid(i) = '1') then
        index_tehb <= std_logic_vector(to_unsigned(i, {index_bitwidth}));
        exit;
      end if;
    end loop;
  end process;

  merge_ins : entity work.{merge_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => tehbOut_ready,
      ins_ready  => ins_ready,
      outs_valid => dataAvailable
    );

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => dataAvailable,
      outs_ready => readyToFork,
      outs_valid => tehbOut_valid,
      ins_ready  => tehbOut_ready,
      ins        => index_tehb,
      outs       => index
    );

  fork_valid : entity work.{fork_name}(arch)
    port map(
      clk           => clk,
      rst           => rst,
      ins_valid     => tehbOut_valid,
      outs_ready(0) => outs_ready,
      outs_ready(1) => index_ready,
      ins_ready     => readyToFork,
      outs_valid(0) => outs_valid,
      outs_valid(1) => index_valid
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_control_merge(name, size, index_bitwidth, data_bitwidth):
    inner_name = f"{name}_inner"

    dependencies = _generate_control_merge_dataless(
        inner_name, size, index_bitwidth)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of control_merge
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- data output channel
    outs       : out std_logic_vector({data_bitwidth} - 1 downto 0);
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
  signal index_internal : std_logic_vector({index_bitwidth} - 1 downto 0);
begin
  control : entity work.{inner_name}
    port map(
      clk         => clk,
      rst         => rst,
      ins_valid   => ins_valid,
      ins_ready   => ins_ready,
      outs_valid  => outs_valid,
      outs_ready  => outs_ready,
      index       => index_internal,
      index_valid => index_valid,
      index_ready => index_ready
    );

  index <= index_internal;
  outs  <= ins(to_integer(unsigned(index_internal)));
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
