import os
import shutil

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports
from generators.support.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.oehb import generate_oehb
from generators.handshake.ofifo import generate_ofifo

def generate_addf(name, options, out_directory):
  data_type = VhdlScalarType(options["data_type"])

  if data_type.bitwidth == 32:
    is_double = False
  elif data_type.bitwidth == 64:
    is_double = True
  else:
    raise ValueError(f"Unsupported bitwidth {data_type.bitwidth}")

  # TODO: There might be a better way to resolve the external dependency
  # Copy the external dependency file to the output directory
  external_dependency = os.path.join(os.path.dirname(__file__),
                                     "external/flopoco_ip_cores.vhd")
  # File is overwritten if it already exists
  shutil.copy(external_dependency, out_directory)

  if data_type.has_extra_signals():
    return _generate_addf_signal_manager(name, data_type, is_double)
  else:
    return _generate_addf(name, is_double)

def _generate_addf(name, is_double):
  if is_double:
    return _generate_addf_double_precision(name)
  else:
    return _generate_addf_single_precision(name)

def _get_latency(is_double):
  return 12 if is_double else 9 # todo

def _generate_addf_single_precision(name):
  join_name = f"{name}_join"
  oehb_name = f"{name}_oehb"
  buff_name = f"{name}_buff"

  dependencies = generate_join(join_name, 2) + \
    generate_oehb(oehb_name, {"data_type": "!handshake.channel<i1>"}) + \
    generate_delay_buffer(buff_name, {"slots": _get_latency(is_double=False) - 1})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addf_single_precision
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(32 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(32 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(32 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of addf_single_precision
architecture arch of {name} is
  signal join_valid : std_logic;
  signal buff_valid, oehb_ready : std_logic;

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(32 + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(32 + 1 downto 0);
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  oehb : entity work.{oehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins(0)     => '0',
      outs    => open
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  ieee2nfloat_lhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.OutputIEEE_32bit(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.FloatingPointAdder(arch)
    port map (
        clk => clk,
        ce  => oehb_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );
end architecture;
"""

  return dependencies + entity + architecture

def _generate_addf_double_precision(name):
  join_name = f"{name}_join"
  oehb_name = f"{name}_oehb"
  buff_name = f"{name}_buff"

  dependencies = generate_join(join_name, 2) + \
    generate_oehb(oehb_name, {"data_type": "!handshake.channel<i1>"}) + \
    generate_delay_buffer(buff_name, {"slots": _get_latency(is_double=True) - 1})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addf_double_precision
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(64 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(64 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(64 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of addf_double_precision
architecture arch of {name} is
  signal join_valid : std_logic;
  signal buff_valid, oehb_ready : std_logic;

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(64 + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(64 + 1 downto 0);
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  oehb : entity work.{oehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins(0)     => '0',
      outs    => open
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  ieee2nfloat_lhs: entity work.InputIEEE_64bit(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.InputIEEE_64bit(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.OutputIEEE_64bit(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.FPAdd_64bit(arch)
    port map (
        clk => clk,
        ce  => oehb_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );
end architecture;
"""

  return dependencies + entity + architecture

def _generate_addf_signal_manager(name, data_type, is_double):
  inner_name = f"{name}_inner"

  dependencies = _generate_addf(inner_name, is_double)

  if "spec" in data_type.extra_signals:
    dependencies += generate_ofifo(f"{name}_spec_ofifo", {
      "num_slots": _get_latency(is_double), # todo: correct?
      "port_types": str({
        "ins": "!handshake.channel<i1>",
        "outs": "!handshake.channel<i1>"
      })
    })

  # Now that the logic depends on the name, this dict is defined inside this function.
  extra_signal_logic = {
    "spec": (
      # First string is for the signal declaration
      """
    signal spec_tfifo_in : std_logic_vector(0 downto 0);
    signal spec_tfifo_out : std_logic_vector(0 downto 0);
""",
      # Second string is for the actual logic
      f"""
    spec_tfifo_in <= lhs_spec or rhs_spec;
    spec_tfifo : entity work.{name}_spec_ofifo(arch)
      port map(
        clk => clk,
        rst => rst,
        ins => spec_tfifo_in,
        ins_valid => transfer,
        ins_ready => open,
        outs => spec_tfifo_out,
        outs_valid => open,
        outs_ready => result_ready
      );
    result_spec <= spec_tfifo_out(0);
""")
  }

  for signal_name in data_type.extra_signals:
    if signal_name not in extra_signal_logic:
      raise ValueError(f"Extra signal {signal_name} is not supported")

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addf signal manager
entity {name} is
  port (
    [EXTRA_SIGNAL_PORTS]
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(32 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(32 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(32 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("lhs", "in"), ("rhs", "in"),
    ("result", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of addf signal manager
architecture arch of {name} is
  signal transfer : std_logic;
  [EXTRA_SIGNAL_SIGNAL_DECLS]
begin
  transfer <= lhs_valid and lhs_ready;

  -- list of logic for supported extra signals
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      rhs => rhs,
      rhs_valid => rhs_valid,
      result_ready => result_ready,
      result => result,
      result_valid => result_valid,
      lhs_ready => lhs_ready,
      rhs_ready => rhs_ready
    );
end architecture;
"""

  architecture = architecture.replace("  [EXTRA_SIGNAL_SIGNAL_DECLS]",
    "\n".join([
      extra_signal_logic[name][0] for name in data_type.extra_signals
    ]))
  architecture = architecture.replace("  [EXTRA_SIGNAL_LOGIC]",
    "\n".join([
      extra_signal_logic[name][1] for name in data_type.extra_signals
    ]))

  return dependencies + entity + architecture

if __name__ == "__main__":
  print(generate_addf("addf", {
    "data_type": "!handshake.channel<i32, [spec: i1]>"
  }, "out/"))
