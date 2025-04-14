from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value
from generators.support.signal_manager.utils.concat import generate_concat_signal_decls_from_ports, ConcatLayout, generate_concat_port_assignments_from_ports
from generators.support.signal_manager.utils.mapping import generate_inner_port_mapping, generate_concat_mappings
from generators.support.signal_manager.utils.types import Port, ArrayPort, ExtraSignals
from generators.support.signal_manager.utils.bbmerge import generate_bbmerge_lacking_spec_statements
from generators.handshake.tehb import generate_tehb
from generators.handshake.merge_notehb import generate_merge_notehb
from generators.handshake.fork import generate_fork


def generate_control_merge(name, params):
  # Number of data input ports
  size = params["size"]

  data_bitwidth = params["data_bitwidth"]
  index_bitwidth = params["index_bitwidth"]

  # List of extra signals for each data input port
  # Each element is a dictionary where key: extra signal name, value: bitwidth
  # e.g., [{"tag0": 8, "spec": 1}, {"tag0": 8}]
  input_extra_signals_list = params["input_extra_signals_list"]
  # e.g., {"tag0": 8, "spec": 1}
  output_extra_signals = params["output_extra_signals"]
  index_extra_signals = params["index_extra_signals"]

  # List of indices of input ports that have spec bit
  # e.g., [0]
  spec_inputs = params["spec_inputs"]

  if output_extra_signals:
    return _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, index_extra_signals, spec_inputs)
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


def _generate_cmerge_index_extra_signal_assignments(index_name: str, index_extra_signals: ExtraSignals) -> str:
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


def _generate_control_merge_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, index_extra_signals, spec_inputs):
  # Declare Ports
  data_in_port: ArrayPort = {
      "name": "ins",
      "bitwidth": data_bitwidth,
      "array": True,
      "size": size,
      "extra_signals_list": input_extra_signals_list
  }
  index_port: Port = {
      "name": "index",
      "bitwidth": index_bitwidth,
      # TODO: Extra signals for index port are not tested
      "extra_signals": index_extra_signals
  }
  data_out_port: Port = {
      "name": "outs",
      "bitwidth": data_bitwidth,
      "extra_signals": output_extra_signals
  }

  # Generate signal manager entity
  entity = generate_entity(
      name,
      [data_in_port],
      [index_port, data_out_port]
  )

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(output_extra_signals)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = _generate_control_merge(
      inner_name, size, index_bitwidth, extra_signals_bitwidth + data_bitwidth)

  # Generate default `spec` bits for inputs that lack them
  lacking_spec_port_decls, lacking_spec_port_assignments = generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, "ins")

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = "\n  ".join(generate_concat_signal_decls_from_ports(
      [data_in_port, data_out_port], extra_signals_bitwidth))

  # Assign inner concatenated signals
  concat_logic = "\n  ".join(generate_concat_port_assignments_from_ports(
      [data_in_port], [data_out_port], concat_layout))

  # Assign index extra signals
  index_extra_signal_assignments = _generate_cmerge_index_extra_signal_assignments(
      "index", index_extra_signals)

  # Map all ports to inner entity:
  #   - Forward concatenated extra signal vectors
  #   - Pass through index port as-is
  mappings = ",\n      ".join(generate_concat_mappings(
      [data_in_port, data_out_port], extra_signals_bitwidth) +
      generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (cmerge)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

  -- Concatenate data and extra signals
  {concat_logic}

  -- Assign index extra signals (if any)
  {index_extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
