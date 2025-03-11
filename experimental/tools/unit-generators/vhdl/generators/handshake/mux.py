from generators.support.utils import generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless, extra_signal_default_values
from generators.handshake.tehb import generate_tehb


def generate_mux(name, params):
  size = params["size"]
  data_bitwidth = params["data_bitwidth"]
  index_bitwidth = params["index_bitwidth"]
  input_extra_signals_list = params["input_extra_signals_list"]
  output_extra_signals = params["output_extra_signals"]
  spec_inputs = params["spec_inputs"]

  if output_extra_signals:
    if data_bitwidth == 0:
      return _generate_mux_signal_manager_dataless(name, size, index_bitwidth, input_extra_signals_list, output_extra_signals, spec_inputs)
    else:
      return _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, spec_inputs)
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


def _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, spec_inputs):
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  # Specify offset for data bitwidth
  extra_signal_mapping = ExtraSignalMapping(offset=data_bitwidth)
  for input_extra_signals in input_extra_signals_list:
    for signal_name, signal_bitwidth in input_extra_signals.items():
      if not extra_signal_mapping.has(signal_name):
        extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate mux for concatenated data and extra signals
  dependencies = _generate_mux(inner_name, size, index_bitwidth, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of mux signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
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

  # Add extra signal ports
  extra_signal_port_decls = []

  # Generate extra signal ports for each input channel (ins_0, ins_1, ...)
  for i in range(size):
    extra_signal_port_decls.append(generate_extra_signal_ports(
        [(f"ins_{i}", "in")], input_extra_signals_list[i]))

  extra_signal_port_decls.append(generate_extra_signal_ports(
      [("outs", "out")], output_extra_signals))

  entity = entity.replace(
      "    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_port_decls))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  -- Concatenated data and extra signals
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  [LACKING_SPEC_INPUT_DECLS]
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"  signal ins_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  architecture = architecture.replace(
      "  [LACKING_SPEC_INPUT_DECLS]",
      "\n".join(lacking_spec_port_decls)
  )

  spec_default_value = extra_signal_default_values["spec"]
  lacking_spec_port_assignments = [
      f"  ins_{i}_spec <= {spec_default_value};" for i in lacking_spec_ports
  ]

  # Concatenate data and extra signals based on extra signal mapping
  ins_conversions = []
  for i in range(size):
    ins_conversions.append(generate_ins_concat_statements(
        f"ins_{i}", f"ins_inner({i})", extra_signal_mapping, data_bitwidth, custom_data_name=f"ins({i})"))
  outs_conversions = generate_outs_concat_statements(
      "outs", "outs_inner", extra_signal_mapping, data_bitwidth)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      "\n".join(lacking_spec_port_assignments) + "\n" +
      "\n".join(ins_conversions) + "\n" + outs_conversions
  )

  return dependencies + entity + architecture


def _generate_mux_signal_manager_dataless(name, size, index_bitwidth, input_extra_signals_list, output_extra_signals, spec_inputs):
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for input_extra_signals in input_extra_signals_list:
    for signal_name, signal_bitwidth in input_extra_signals.items():
      if not extra_signal_mapping.has(signal_name):
        extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate mux for concatenated extra signals
  dependencies = _generate_mux(inner_name, size, index_bitwidth, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of mux signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
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

  # Add extra signal ports
  extra_signal_port_decls = []

  # Generate extra signal ports for each input channel (ins_0, ins_1, ...)
  for i in range(size):
    extra_signal_port_decls.append(generate_extra_signal_ports(
        [(f"ins_{i}", "in")], input_extra_signals_list[i]))

  extra_signal_port_decls.append(generate_extra_signal_ports(
      [("outs", "out")], output_extra_signals))

  entity = entity.replace(
      "    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_port_decls))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  [LACKING_SPEC_INPUT_DECLS]
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"  signal ins_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  architecture = architecture.replace(
      "  [LACKING_SPEC_INPUT_DECLS]",
      "\n".join(lacking_spec_port_decls)
  )

  spec_default_value = extra_signal_default_values["spec"]
  lacking_spec_port_assignments = [
      f"  ins_{i}_spec <= {spec_default_value};" for i in lacking_spec_ports
  ]

  # Concatenate extra signals based on extra signal mapping
  ins_conversions = []
  for i in range(size):
    ins_conversions.append(generate_ins_concat_statements_dataless(
        f"ins_{i}", f"ins_inner({i})", extra_signal_mapping))
  outs_conversions = generate_outs_concat_statements_dataless(
      "outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      "\n".join(lacking_spec_port_assignments) + "\n" +
      "\n".join(ins_conversions) + "\n" + outs_conversions
  )

  return dependencies + entity + architecture
