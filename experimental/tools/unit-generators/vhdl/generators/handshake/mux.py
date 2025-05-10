from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat, generate_slice, generate_handshake_forwarding, generate_signal_wise_forwarding
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

  # Concatenate ins data and extra signals to create ins_inner
  assignments, decls = generate_concat(
      "ins", data_bitwidth, "ins_inner", concat_layout, size)
  concat_assignments.extend(assignments)
  # Declare ins_inner data
  concat_decls.extend(decls["ins_inner"])

  # Forward ins handshake to ins_inner
  assignments, decls = generate_handshake_forwarding(
      "ins", "ins_inner", size)
  concat_assignments.extend(assignments)
  # Declare ins_inner handshake
  concat_decls.extend(decls["ins_inner"])

  return "\n  ".join(concat_assignments), "\n  ".join(concat_decls)


def _generate_slice(data_bitwidth: int, concat_layout: ConcatLayout) -> tuple[str, str]:
  slice_decls = []
  slice_assignments = []

  # Slice outs_inner_concat to create outs_inner data and extra signals
  assignments, decls = generate_slice(
      "outs_inner_concat", "outs_inner", data_bitwidth, concat_layout)
  slice_assignments.extend(assignments)
  # Declare both outs_inner_concat data signal and outs_inner data and extra signals
  slice_decls.extend(decls["outs_inner_concat"])
  slice_decls.extend(decls["outs_inner"])

  # Forward outs_inner_concat handshake to outs_inner
  assignments, decls = generate_handshake_forwarding(
      "outs_inner_concat", "outs_inner")
  slice_assignments.extend(assignments)
  # Declare both outs_inner_concat handshake and outs_inner handshake
  slice_decls.extend(decls["outs_inner_concat"])
  slice_decls.extend(decls["outs_inner"])

  return "\n  ".join(slice_assignments), "\n  ".join(slice_decls)


def _generate_forwarding(extra_signals: ExtraSignals) -> str:
  forwarding_assignments = []
  for signal_name, signal_bitwidth in extra_signals.items():
    # Signal-wise forwarding of extra signals from ins_inner and outs_inner to outs
    assignments, _ = generate_signal_wise_forwarding(
        ["index", "outs_inner"], ["outs"], signal_name, signal_bitwidth)
    forwarding_assignments.extend(assignments)

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
