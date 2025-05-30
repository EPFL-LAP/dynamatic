from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_signal_wise_forwarding
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl
from generators.handshake.tehb import generate_tehb
from generators.support.signal_manager.utils.types import ExtraSignals


def generate_mux(name, params):
    # Number of data input ports
    size = params["size"]

    data_bitwidth = params["data_bitwidth"]
    index_bitwidth = params["index_bitwidth"]

    # e.g., {"tag0": 8, "spec": 1}
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals)
    elif data_bitwidth == 0:
        return _generate_mux_dataless(name, size, index_bitwidth)
    else:
        return _generate_mux(name, size, index_bitwidth, data_bitwidth)


def _generate_mux(name, size, index_bitwidth, data_bitwidth):
    tehb_name = f"{name}_tehb"

    dependencies = generate_tehb(tehb_name, {"bitwidth": data_bitwidth})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

-- Entity of mux
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of mux
architecture arch of {name} is
  signal tehb_ins                       : std_logic_vector({data_bitwidth} - 1 downto 0);
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
begin
  process (ins, ins_valid, outs_ready, index, index_valid, tehb_ins_ready)
    variable selectedData                   : std_logic_vector({data_bitwidth} - 1 downto 0);
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in {size} - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready    <= (not index_valid) or (selectedData_valid and tehb_ins_ready);
    tehb_ins       <= selectedData;
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tehb_ins,
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_mux_dataless(name, size, index_bitwidth):
    tehb_name = f"{name}_tehb"

    dependencies = generate_tehb(tehb_name, {"bitwidth": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

-- Entity of mux_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of mux_dataless
architecture arch of {name} is
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
begin
  process (ins_valid, outs_ready, index, index_valid, tehb_ins_ready)
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData_valid := '0';

    for i in {size} - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;

      if indexEqual and index_valid and ins_valid(i) then
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready    <= (not index_valid) or (selectedData_valid and tehb_ins_ready);
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_concat(data_bitwidth: int, concat_layout: ConcatLayout, size: int) -> tuple[str, str]:
    concat_decls = []
    concat_assignments = []

    # Declare ins_inner channel
    # Example:
    # signal ins_inner : data_array(1 downto 0)(32 downto 0);
    # signal ins_inner_valid : std_logic_vector(1 downto 0);
    # signal ins_inner_ready : std_logic_vector(1 downto 0);
    concat_decls.extend(create_internal_channel_decl({
        "name": "ins_inner",
        "bitwidth": data_bitwidth + concat_layout.total_bitwidth,
        "size": size
    }))

    # Concatenate ins data and extra signals to create ins_inner
    # Example:
    # ins_inner(0)(32 - 1 downto 0) <= ins(0);
    # ins_inner(0)(32 downto 32) <= ins_0_spec;
    # ins_inner(1)(32 - 1 downto 0) <= ins(1);
    # ins_inner(1)(32 downto 32) <= ins_1_spec;
    # ins_inner_valid <= ins_valid;
    # ins_ready <= ins_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "ins", data_bitwidth, "ins_inner", concat_layout, size))

    return "\n  ".join(concat_assignments), "\n  ".join(concat_decls)


def _generate_slice(data_bitwidth: int, concat_layout: ConcatLayout) -> tuple[str, str]:
    slice_decls = []
    slice_assignments = []

    # Declare both outs_inner_concat and outs_inner channels
    # Example:
    # signal outs_inner_concat : std_logic_vector(32 downto 0);
    # signal outs_inner_concat_valid : std_logic;
    # signal outs_inner_concat_ready : std_logic;
    slice_decls.extend(create_internal_channel_decl({
        "name": "outs_inner_concat",
        "bitwidth": data_bitwidth + concat_layout.total_bitwidth
    }))

    # Example:
    # signal outs_inner : std_logic_vector(31 downto 0);
    # signal outs_inner_valid : std_logic;
    # signal outs_inner_ready : std_logic;
    # signal outs_inner_spec : std_logic_vector(0 downto 0);
    slice_decls.extend(create_internal_channel_decl({
        "name": "outs_inner",
        "bitwidth": data_bitwidth,
        "extra_signals": concat_layout.extra_signals
    }))

    # Slice outs_inner_concat to create outs_inner data and extra signals
    # Example:
    # outs_inner <= outs_inner_concat(32 - 1 downto 0);
    # outs_inner_spec <= outs_inner_concat(32 downto 32);
    # outs_inner_valid <= outs_inner_concat_valid;
    # outs_inner_concat_ready <= outs_inner_ready;
    slice_assignments.extend(generate_slice_and_handshake(
        "outs_inner_concat", "outs_inner", data_bitwidth, concat_layout))

    return "\n  ".join(slice_assignments), "\n  ".join(slice_decls)


def _generate_forwarding(extra_signals: ExtraSignals) -> str:
    forwarding_assignments = []

    # Signal-wise forwarding of extra signals from ins_inner and outs_inner to outs
    # Example:
    # outs_spec <= index_spec or outs_inner_spec;
    for signal_name in extra_signals:
        forwarding_assignments.extend(generate_signal_wise_forwarding(
            ["index", "outs_inner"], ["outs"], signal_name))

    return "\n  ".join(forwarding_assignments)


def _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals):
    # Generate signal manager entity
    entity = generate_entity(
        name,
        [{
            "name": "ins",
            "bitwidth": data_bitwidth,
            "size": size,
            "extra_signals": extra_signals
        }, {
            "name": "index",
            "bitwidth": index_bitwidth,
            # TODO: Extra signals for index port are not tested
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": data_bitwidth,
            "extra_signals": extra_signals
        }]
    )

    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_mux(inner_name, size, index_bitwidth,
                          extra_signals_bitwidth + data_bitwidth)

    concat_assignments, concat_decls = _generate_concat(
        data_bitwidth, concat_layout, size)
    slice_assignments, slice_decls = _generate_slice(
        data_bitwidth, concat_layout)
    forwarding_assignments = _generate_forwarding(extra_signals)

    architecture = f"""
-- Architecture of signal manager (mux)
architecture arch of {name} is
  {concat_decls}
  {slice_decls}
begin
  -- Concatenate data and extra signals
  {concat_assignments}
  {slice_assignments}

  -- Forwarding logic
  {forwarding_assignments}

  outs <= outs_inner;
  outs_valid <= outs_inner_valid;
  outs_inner_ready <= outs_ready;

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_inner_valid,
      ins_ready => ins_inner_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready,
      outs => outs_inner_concat,
      outs_valid => outs_inner_concat_valid,
      outs_ready => outs_inner_concat_ready
    );
end architecture;
"""

    return inner + entity + architecture
