from generators.support.utils import generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.support.logic import generate_or_n
from generators.support.eager_fork_register_block import generate_eager_fork_register_block


def generate_fork(name, params):
  bitwidth = params["bitwidth"]
  size = params["size"]
  extra_signals = params["extra_signals"]

  if extra_signals:
    if bitwidth == 0:
      return _generate_fork_signal_manager_dataless(name, size, extra_signals)
    else:
      return _generate_fork_signal_manager(name, size, bitwidth, extra_signals)
  elif bitwidth == 0:
    return _generate_fork_dataless(name, size)
  else:
    return _generate_fork(name, size, bitwidth)


def _generate_fork_dataless(name, size):
  or_n_name = f"{name}_or_n"
  regblock_name = f"{name}_regblock"

  dependencies = \
      generate_or_n(or_n_name, {"size": size}) + \
      generate_eager_fork_register_block(regblock_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fork_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of fork_dataless
architecture arch of {name} is
  signal blockStopArray : std_logic_vector({size} - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.{or_n_name}
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in {size} - 1 downto 0 generate
    regblock : entity work.{regblock_name}(arch)
      port map(
        -- inputs
        clk          => clk,
        rst          => rst,
        ins_valid    => ins_valid,
        outs_ready   => outs_ready(i),
        backpressure => backpressure,
        -- outputs
        outs_valid => outs_valid(i),
        blockStop  => blockStopArray(i)
      );
  end generate;

end architecture;
"""

  return dependencies + entity + architecture


def _generate_fork(name, size, bitwidth):
  inner_name = f"{name}_inner"

  dependencies = _generate_fork_dataless(inner_name, size)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of fork
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of fork
architecture arch of {name} is
begin
  control : entity work.{inner_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (ins)
  begin
    for i in 0 to {size} - 1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;
"""

  return dependencies + entity + architecture


def _generate_fork_signal_manager(name, size, bitwidth, extra_signals):
  inner_name = f"{name}_inner"

  extra_signal_mapping = ExtraSignalMapping(bitwidth)
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_fork(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of fork signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_need_ports = [("ins", "in")]
  for i in range(size):
    extra_signal_need_ports.append((f"outs_{i}", "out"))
  extra_signal_ports = generate_extra_signal_ports(
      extra_signal_need_ports, extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of fork signal manager
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins_inner,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_conversion = generate_ins_concat_statements(
      "ins", "ins_inner", extra_signal_mapping, bitwidth)
  outs_conversion = []
  for i in range(size):
    outs_conversion.append(generate_outs_concat_statements(
        f"outs_{i}", f"outs_inner({i})", extra_signal_mapping, bitwidth, custom_data_name=f"outs({i})"))

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + "\n" + "\n".join(outs_conversion)
  )

  return dependencies + entity + architecture


def _generate_fork_signal_manager_dataless(name, size, extra_signals):
  inner_name = f"{name}_inner"

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_fork(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of fork signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_need_ports = [("ins", "in")]
  for i in range(size):
    extra_signal_need_ports.append((f"outs_{i}", "out"))
  extra_signal_ports = generate_extra_signal_ports(
      extra_signal_need_ports, extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of fork signal manager dataless
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : data_array({size} - 1 downto 0)({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins_inner,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_conversion = generate_ins_concat_statements_dataless(
      "ins", "ins_inner", extra_signal_mapping)
  outs_conversion = []
  for i in range(size):
    outs_conversion.append(generate_outs_concat_statements_dataless(
        f"outs_{i}", f"outs_inner({i})", extra_signal_mapping))

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + "\n".join(outs_conversion)
  )

  return dependencies + entity + architecture
