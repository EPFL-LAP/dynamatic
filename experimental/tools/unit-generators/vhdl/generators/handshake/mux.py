import ast

from generators.support.utils import VhdlScalarType
from generators.support.array import generate_2d_array
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
  array_name = f"{name}_array"

  dependencies = generate_tehb(tehb_name, {
    "port_types": str({
      "ins": f"!handshake.channel<i{data_bitwidth}>",
      "outs": f"!handshake.channel<i{data_bitwidth}>",
    })
  }) + generate_2d_array(array_name, size, data_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.{array_name}.all;

-- Entity of mux
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  {array_name};
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
  array_name = f"{name}_array"
  array_fullwidth_name = f"{name}_array_fullwidth"

  outs_type = VhdlScalarType(port_types["outs"])
  ins_types = []
  index_type = VhdlScalarType(port_types["index"])

  bitwidth = outs_type.bitwidth
  index_bitwidth = index_type.bitwidth

  extra_signal_bit_map = {}
  occupied_bits = bitwidth
  for i in range(size):
    ins_i_name = f"ins_{i}"
    ins_i_type = VhdlScalarType(port_types[ins_i_name])
    ins_types.append(ins_i_type)

    for signal_name, signal_bitwidth in ins_i_type.extra_signals.items():
      if signal_name not in extra_signal_bit_map:
        extra_signal_bit_map[signal_name] = (
          occupied_bits + signal_bitwidth - 1,
          occupied_bits
        )
        occupied_bits += signal_bitwidth

  full_bitwidth = occupied_bits

  dependencies = _generate_mux(inner_name, size, index_bitwidth, full_bitwidth) + \
    generate_2d_array(array_name, size, bitwidth) + \
    generate_2d_array(array_fullwidth_name, size, bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.{array_name}.all;
use work.{array_fullwidth_name}.all;

-- Entity of mux signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- data input channels
    ins       : in  {array_name};
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
  extra_signal_ports = []
  for i in range(size):
    for signal_name, signal_bitwidth in ins_types[i].extra_signals.items():
      extra_signal_ports.append(
        f"ins_{i}_{signal_name} : in std_logic_vector({signal_bitwidth - 1} downto 0);"
      )
  for signal_name, (msb, lsb) in extra_signal_bit_map.items():
    extra_signal_ports.append(
      f"outs_{signal_name} : out std_logic_vector({msb - lsb} downto {0});"
    )
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_ports))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  signal ins_inner : {array_fullwidth_name};
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
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

  default_values = {
    "spec": "'0'",
  }

  ins_conversions = []
  for i in range(size):
    ins_inner = ["ins"]
    for signal_name in extra_signal_bit_map.keys():
      if signal_name in ins_types[i].extra_signals:
        ins_inner.append(f"ins_{i}_{signal_name}")
      else:
        ins_inner.append(default_values[signal_name])
    ins_inner.reverse()
    ins_conversions.append(f"  ins_inner({i}) <= {" & ".join(ins_inner)};")

  outs_conversions = [
    f"  outs <= outs_inner({bitwidth} - 1 downto 0);"
  ]
  for signal_name, (msb, lsb) in extra_signal_bit_map.items():
    outs_conversions.append(
      f"  outs_{signal_name} <= outs_inner({msb} downto {lsb});"
    )

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    "\n".join(ins_conversions + outs_conversions)
  )

  return dependencies + entity + architecture

def _generate_mux_signal_manager_dataless(name, size, port_types):
  inner_name = f"{name}_inner"
  array_name = f"{name}_array"

  ins_types = []
  index_type = VhdlScalarType(port_types["index"])

  bitwidth = 0
  index_bitwidth = index_type.bitwidth

  extra_signal_bit_map = {}
  occupied_bits = bitwidth
  for i in range(size):
    ins_i_name = f"ins_{i}"
    ins_i_type = VhdlScalarType(port_types[ins_i_name])
    ins_types.append(ins_i_type)

    for signal_name, signal_bitwidth in ins_i_type.extra_signals.items():
      if signal_name not in extra_signal_bit_map:
        extra_signal_bit_map[signal_name] = (
          occupied_bits + signal_bitwidth - 1,
          occupied_bits
        )
        occupied_bits += signal_bitwidth

  full_bitwidth = occupied_bits

  dependencies = _generate_mux(inner_name, size, index_bitwidth, full_bitwidth) + \
    generate_2d_array(array_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.{array_name}.all;

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
  extra_signal_ports = []
  for i in range(size):
    for signal_name, signal_bitwidth in ins_types[i].extra_signals.items():
      extra_signal_ports.append(
        f"ins_{i}_{signal_name} : in std_logic_vector({signal_bitwidth - 1} downto 0);"
      )
  for signal_name, (msb, lsb) in extra_signal_bit_map.items():
    extra_signal_ports.append(
      f"outs_{signal_name} : out std_logic_vector({msb - lsb} downto {0});"
    )
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", "\n".join(extra_signal_ports))

  architecture = f"""
-- Architecture of mux signal manager
architecture arch of {name} is
  signal ins_inner : {array_name};
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
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

  default_values = {
    "spec": "'0'",
  }

  ins_conversions = []
  for i in range(size):
    ins_inner = []
    for signal_name in extra_signal_bit_map.keys():
      if signal_name in ins_types[i].extra_signals:
        ins_inner.append(f"ins_{i}_{signal_name}")
      else:
        ins_inner.append(default_values[signal_name])
    ins_inner.reverse()
    ins_conversions.append(f"  ins_inner({i}) <= {" & ".join(ins_inner)};")

  outs_conversions = []
  for signal_name, (msb, lsb) in extra_signal_bit_map.items():
    outs_conversions.append(
      f"  outs_{signal_name} <= outs_inner({msb} downto {lsb});"
    )

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    "\n".join(ins_conversions + outs_conversions)
  )

  return dependencies + entity + architecture