from generators.support.utils import VhdlScalarType, generate_extra_signal_ports


def generate_extsi(name, params):
  port_types = params["port_types"]
  ins_type = VhdlScalarType(port_types["ins"])
  outs_type = VhdlScalarType(port_types["outs"])

  if ins_type.has_extra_signals():
    return _generate_extsi_signal_manager(name, ins_type, outs_type)
  else:
    return _generate_extsi(name, ins_type.bitwidth, outs_type.bitwidth)


def _generate_extsi(name, input_bitwidth, output_bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({input_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({output_bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of extsi
architecture arch of {name} is
begin
  outs({output_bitwidth} - 1 downto {input_bitwidth}) <= ({output_bitwidth} - {input_bitwidth} - 1 downto 0 => ins({input_bitwidth} - 1));
  outs({input_bitwidth} - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;
"""

  return entity + architecture


extra_signal_logic = {
    "spec": """
  outs_spec <= ins_spec;
"""
}


def _generate_extsi_signal_manager(name, ins_type, outs_type):
  inner_name = f"{name}_inner"

  input_bitwidth = ins_type.bitwidth
  output_bitwidth = outs_type.bitwidth

  dependencies = _generate_extsi(inner_name, input_bitwidth, output_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins       : in  std_logic_vector({input_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({output_bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
      ("ins", "in"),
      ("outs", "out")
  ], ins_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of extsi signal manager
architecture arch of {name} is
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  architecture = architecture.replace("  [EXTRA_SIGNAL_LOGIC]", "\n".join([
      extra_signal_logic[name] for name in ins_type.extra_signals
  ]))

  return dependencies + entity + architecture
