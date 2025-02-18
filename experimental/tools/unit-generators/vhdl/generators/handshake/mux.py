import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_lacking_extra_signal_decls, generate_lacking_extra_signal_assignments, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.handshake.tehb import generate_tehb

def generate_mux(name, params):
  size = int(params["size"])
  port_types = ast.literal_eval(params["port_types"])
  outs_type = VhdlScalarType(port_types["outs"])
  index_type = VhdlScalarType(port_types["index"])

  if outs_type.has_extra_signals():
    if outs_type.is_channel():
      return _generate_mux_signal_manager(name, size, port_types)
    else:
      return _generate_mux_signal_manager_dataless(name, size, port_types)
  elif outs_type.is_channel():
    return _generate_mux(name, size, index_type.bitwidth, outs_type.bitwidth)
  else:
    return _generate_mux_dataless(name, size, index_type.bitwidth)

def _generate_mux(name, size, index_bitwidth, data_bitwidth):
  tehb_name = f"{name}_tehb"

  dependencies = generate_tehb(tehb_name, {
    "port_types": str({
      "ins": f"!handshake.channel<i{data_bitwidth}>",
      "outs": f"!handshake.channel<i{data_bitwidth}>",
    })
  })

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

  dependencies = generate_tehb(tehb_name, {
    "port_types": str({
      "ins": f"!handshake.control<>",
      "outs": f"!handshake.control<>",
    })
  })

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

def _generate_mux_signal_manager(name, size, port_types):
  inner_name = f"{name}_inner"

  outs_type = VhdlScalarType(port_types["outs"])
  ins_types = []
  index_type = VhdlScalarType(port_types["index"])

  bitwidth = outs_type.bitwidth
  index_bitwidth = index_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for i in range(size):
    ins_i_name = f"ins_{i}"
    ins_i_type = VhdlScalarType(port_types[ins_i_name])
    ins_types.append(ins_i_type)

    for signal_name, signal_bitwidth in ins_i_type.extra_signals.items():
      if not extra_signal_mapping.has(signal_name):
        extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

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
    ins       : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_port_decls = []
  for i in range(size):
    extra_signal_port_decls.append(generate_extra_signal_ports([(f"ins_{i}", "in")], ins_types[i].extra_signals))
  extra_signal_port_decls.append(generate_extra_signal_ports([("outs", "out")], extra_signal_mapping.to_extra_signals()))
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_port_decls))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  [LACKING_EXTRA_SIGNAL_DECLS]
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

  architecture = architecture.replace(
    "  [LACKING_EXTRA_SIGNAL_DECLS]",
    generate_lacking_extra_signal_decls("ins", ins_types, extra_signal_mapping)
  )

  lacking_extra_signal_assignments = generate_lacking_extra_signal_assignments("ins", ins_types, extra_signal_mapping)

  ins_conversions = []
  for i in range(size):
    ins_conversions.append(generate_ins_concat_statements(f"ins_{i}", f"ins_inner({i})", extra_signal_mapping, bitwidth, custom_data_name=f"ins({i})"))

  outs_conversions = generate_outs_concat_statements("outs", "outs_inner", extra_signal_mapping, bitwidth)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    lacking_extra_signal_assignments + "\n" + \
    "\n".join(ins_conversions) + "\n" + outs_conversions
  )

  return dependencies + entity + architecture

def _generate_mux_signal_manager_dataless(name, size, port_types):
  inner_name = f"{name}_inner"

  ins_types = []
  index_type = VhdlScalarType(port_types["index"])

  index_bitwidth = index_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping()
  for i in range(size):
    ins_i_name = f"ins_{i}"
    ins_i_type = VhdlScalarType(port_types[ins_i_name])
    ins_types.append(ins_i_type)

    for signal_name, signal_bitwidth in ins_i_type.extra_signals.items():
      if not extra_signal_mapping.has(signal_name):
        extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

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
  for i in range(size):
    extra_signal_port_decls.append(generate_extra_signal_ports([(f"ins_{i}", "in")], ins_types[i].extra_signals))
  extra_signal_port_decls.append(generate_extra_signal_ports([("outs", "out")], extra_signal_mapping.to_extra_signals()))
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_port_decls))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  [LACKING_EXTRA_SIGNAL_DECLS]
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

  architecture = architecture.replace(
    "  [LACKING_EXTRA_SIGNAL_DECLS]",
    generate_lacking_extra_signal_decls("ins", ins_types, extra_signal_mapping)
  )

  lacking_extra_signal_assignments = generate_lacking_extra_signal_assignments("ins", ins_types, extra_signal_mapping)

  ins_conversions = []
  for i in range(size):
    ins_conversions.append(generate_ins_concat_statements_dataless(f"ins_{i}", f"ins_inner({i})", extra_signal_mapping))

  outs_conversions = generate_outs_concat_statements_dataless("outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    lacking_extra_signal_assignments + "\n" + \
    "\n".join(ins_conversions) + "\n" + outs_conversions
  )

  return dependencies + entity + architecture