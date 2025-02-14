import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless

def generate_oehb(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])

  if data_type.has_extra_signals():
    if data_type.is_channel():
      return _generate_oehb_signal_manager(name, data_type)
    else:
      return _generate_oehb_signal_manager_dataless(name, data_type)
  elif data_type.is_channel():
    return _generate_oehb(name, data_type.bitwidth)
  else:
    return _generate_oehb_dataless(name)

def _generate_oehb_dataless(name):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of oehb_dataless
architecture arch of {name} is
  signal outputValid : std_logic;
begin
  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outputValid <= '0';
      else
        outputValid <= ins_valid or (outputValid and not outs_ready);
      end if;
    end if;
  end process;

  ins_ready  <= not outputValid or outs_ready;
  outs_valid <= outputValid;
end architecture;
"""

  return entity + architecture

def _generate_oehb(name, bitwidth):
  inner_name = f"{name}_inner"

  dependencies = _generate_oehb_dataless(inner_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of oehb
architecture arch of oehb is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.{inner_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outs <= (others => '0');
      elsif (regEn) then
        outs <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  regEn     <= inputReady and ins_valid;
end architecture;
"""

  return dependencies + entity + architecture

def _generate_oehb_signal_manager(name, data_type):
  inner_name = f"{name}_inner"

  bitwidth = data_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for signal_name, signal_type in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_oehb(inner_name, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of oehb signal manager
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
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

  ins_conversion = generate_ins_concat_statements("ins", "ins_inner", extra_signal_mapping, bitwidth)
  outs_conversion = generate_outs_concat_statements("outs", "outs_inner", extra_signal_mapping, bitwidth)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture

def _generate_oehb_signal_manager_dataless(name, data_type):
  inner_name = f"{name}_inner"

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_oehb(inner_name, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of oehb signal manager dataless
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
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

  ins_conversion = generate_ins_concat_statements_dataless("ins", "ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless("outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture
