from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import generate_concat_signal_decls_from_ports, ConcatLayout, generate_concat_port_assignments_from_ports
from generators.support.signal_manager.utils.mapping import generate_inner_port_mapping, generate_concat_mappings
from generators.support.signal_manager.utils.types import Port, ArrayPort
from generators.handshake.tehb import generate_tehb


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


def _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, extra_signals):
  # Declare Ports
  data_in_port: ArrayPort = {
      "name": "ins",
      "bitwidth": data_bitwidth,
      "array": True,
      "size": size,
      "extra_signals": extra_signals
  }
  index_port: Port = {
      "name": "index",
      "bitwidth": index_bitwidth,
      # TODO: Extra signals for index port are not tested
      "extra_signals": extra_signals
  }
  data_out_port: Port = {
      "name": "outs",
      "bitwidth": data_bitwidth,
      "extra_signals": extra_signals
  }

  # Generate signal manager entity
  entity = generate_entity(
      name,
      [data_in_port, index_port],
      [data_out_port]
  )

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = _generate_mux(inner_name, size, index_bitwidth,
                        extra_signals_bitwidth + data_bitwidth)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = "\n  ".join(generate_concat_signal_decls_from_ports(
      [data_in_port, data_out_port], extra_signals_bitwidth))

  # Assign inner concatenated signals
  concat_logic = "\n  ".join(generate_concat_port_assignments_from_ports(
      [data_in_port], [data_out_port], concat_layout))

  # Map all ports to inner entity:
  #   - Forward concatenated extra signal vectors
  #   - Pass through index port as-is
  mappings = ",\n      ".join(generate_concat_mappings(
      [data_in_port, data_out_port], extra_signals_bitwidth) +
      generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (mux)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
