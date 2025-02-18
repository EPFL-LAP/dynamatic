import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_lacking_extra_signal_decls, generate_lacking_extra_signal_assignments, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.handshake.tehb import generate_tehb
from generators.support.merge_notehb import generate_merge_notehb
from generators.handshake.fork import generate_fork

def generate_control_merge(name, params):
  size = int(params["size"])
  port_types = ast.literal_eval(params["port_types"])
  outs_type = VhdlScalarType(port_types["outs"])
  index_type = VhdlScalarType(port_types["index"])

  if outs_type.has_extra_signals():
    if outs_type.is_channel():
      return _generate_control_merge_signal_manager(name, size, port_types)
    else:
      return _generate_control_merge_signal_manager_dataless(name, size, port_types)
  elif outs_type.is_channel():
    return _generate_control_merge(name, size, index_type.bitwidth, outs_type.bitwidth)
  else:
    return _generate_control_merge_dataless(name, size, index_type.bitwidth)

def _generate_control_merge_dataless(name, size, index_bitwidth):
  merge_name = f"{name}_merge"
  tehb_name = f"{name}_tehb"
  fork_name = f"{name}_fork"

  dependencies = generate_merge_notehb(merge_name, size) + \
    generate_tehb(tehb_name, {
      "port_types": str({
        "ins": f"!handshake.channel<i{index_bitwidth}>",
        "outs": f"!handshake.channel<i{index_bitwidth}>"
      })
    }) + \
    generate_fork(fork_name, {
      "size": "2",
      "port_types": str({
        "ins": f"!handshake.control<>",
        "outs_0": f"!handshake.control<>",
        "outs_1": f"!handshake.control<>"
      })
    })

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

  dependencies = _generate_control_merge_dataless(inner_name, size, index_bitwidth)

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

def _generate_control_merge_signal_manager(name, size, port_types):
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

  dependencies = _generate_control_merge(inner_name, size, index_bitwidth, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of control merge signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- data input channels
    ins       : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
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
-- Architecture of control merge signal manager
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

def _generate_control_merge_signal_manager_dataless(name, size, port_types):
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

  dependencies = _generate_control_merge(inner_name, size, index_bitwidth, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of control merge signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- data input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
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
-- Architecture of control merge signal manager
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
