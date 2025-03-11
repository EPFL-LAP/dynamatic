from generators.support.utils import generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.handshake.merge_notehb import generate_merge_notehb
from generators.handshake.tehb import generate_tehb


def generate_merge(name, params):
  size = params["size"]
  bitwidth = params["bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    if bitwidth == 0:
      return _generate_merge_signal_manager_dataless(name, size, extra_signals)
    else:
      return _generate_merge_signal_manager(name, size, bitwidth, extra_signals)
  elif bitwidth == 0:
    return _generate_merge_dataless(name, size)
  else:
    return _generate_merge(name, size, bitwidth)


def _generate_merge_dataless(name, size):
  inner_name = f"{name}_inner"
  tehb_name = f"{name}_tehb"

  dependencies = generate_merge_notehb(inner_name, {"size": size}) + \
      generate_tehb(tehb_name, {"size": 0})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of merge_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of merge_dataless
architecture arch of {name} is
  signal tehb_pvalid : std_logic;
  signal tehb_ready  : std_logic;
begin
  merge_ins : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready
    );
end architecture;
"""

  return dependencies + entity + architecture


def _generate_merge(name, size, bitwidth):
  inner_name = f"{name}_inner"
  tehb_name = f"{name}_tehb"

  dependencies = \
      generate_merge_notehb(inner_name, {
          "size": size,
          "bitwidth": bitwidth,
      }) + \
      generate_tehb(tehb_name, {"bitwidth": bitwidth})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of merge
architecture arch of {name} is
  signal tehb_data_in : std_logic_vector({bitwidth} - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin

  merge_ins : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs       => tehb_data_in,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;
"""

  return dependencies + entity + architecture


def _generate_merge_signal_manager(name, size, bitwidth, extra_signals):
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  # Specify offset for data bitwidth
  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate merge for concatenated data and extra signals
  dependencies = _generate_merge(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channels
    ins       : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_need_ports = [("outs", "out")]
  for i in range(size):
    extra_signal_need_ports.append((f"ins_{i}", "in"))
  extra_signal_ports = generate_extra_signal_ports(
      extra_signal_need_ports, extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of merge signal manager
architecture arch of {name} is
  -- Concatenated data and extra signals
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
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
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  # Concatenate data and extra signals based on extra signal mapping
  ins_conversion = []
  for i in range(size):
    ins_conversion.append(generate_ins_concat_statements(
        f"ins_{i}", f"ins_inner({i})", extra_signal_mapping, bitwidth, custom_data_name=f"ins({i})"))
  outs_conversion = generate_outs_concat_statements(
      "outs", "outs_inner", extra_signal_mapping, bitwidth)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      "\n".join(ins_conversion) + "\n" + outs_conversion
  )

  return dependencies + entity + architecture


def _generate_merge_signal_manager_dataless(name, size, extra_signals):
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate merge for concatenated extra signals
  dependencies = _generate_merge(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_need_ports = [("outs", "out")]
  for i in range(size):
    extra_signal_need_ports.append((f"ins_{i}", "in"))
  extra_signal_ports = generate_extra_signal_ports(
      extra_signal_need_ports, extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of merge signal manager dataless
architecture arch of {name} is
  -- Concatenated extra signals
  signal ins_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
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
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  # Concatenate extra signals based on extra signal mapping
  ins_conversion = []
  for i in range(size):
    ins_conversion.append(generate_ins_concat_statements_dataless(
        f"ins_{i}", f"ins_inner({i})", extra_signal_mapping))
  outs_conversion = generate_outs_concat_statements_dataless(
      "outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      "\n".join(ins_conversion) + "\n" + outs_conversion
  )

  return dependencies + entity + architecture
