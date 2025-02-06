from generators.support.utils import VhdlScalarType
from generators.handshake.join import generate_join

# todo: move to somewhere else (like utils.py)
def generate_extra_signal_ports(ports, extra_signals):
  return "    -- extra signal ports\n" + "\n".join([
    "\n".join([
      f"    {port}_{name} : {inout} std_logic_vector({bitwidth - 1} downto 0);"
      for name, bitwidth in extra_signals.items()
    ])
    for port, inout in ports
  ])

def generate_addf(name, options):
  data_type = VhdlScalarType(options["data_type"])

  if data_type.bitwidth == 32:
    is_double = False
  elif data_type.bitwidth == 64:
    is_double = True
  else:
    raise ValueError(f"Unsupported bitwidth {data_type.bitwidth}")

  if data_type.has_extra_signals():
    return _generate_addf_signal_manager(name, is_double)
  else:
    return _generate_addf(name)

def _generate_addf(name, is_double, export_transfer=False):
  if is_double:
    return _generate_addf_single_precision(name, export_transfer)
  else:
    return _generate_addf_double_precision(name, export_transfer)

def _get_latency(is_double):
  return 12 if is_double else 9

def _generate_addf_single_precision(name, export_transfer=False):
  join_name = f"{name}_join"
  oehb_name = f"{name}_oehb"
  buff_name = f"{name}_buff"
  ieee2nfloat_name = f"{name}_ieee2nfloat"
  nfloat2ieee_name = f"{name}_nfloat2ieee"
  floating_point_adder_name = f"{name}_floating_point_adder"
  dependencies = generate_join(join_name, {"size": 2}) + \
    generate_oehb(oehb_name, {"data_type": "!handshake.channel<i1>"}) + \
    generate_delay_buffer(buff_name, {"slots": _get_latency(is_double=False) - 1}) + \
    generate_input_ieee_32bit(ieee2nfloat_name) + \
    generate_output_ieee_32bit(nfloat2ieee_name) + \
    generate_floating_point_adder(floating_point_adder_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
  port (
    [POSSIBLE_TRANSFER]
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
  [POSSIBLE_TRANSFER]

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

  ieee2nfloat_lhs: entity work.{ieee2nfloat_name}(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.{ieee2nfloat_name}(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.{nfloat2ieee_name}(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.{floating_point_adder_name}(arch)
    port map (
        clk => clk,
        ce  => oehb_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );
end architecture;
"""

  if export_transfer:
    entity = entity.replace("[POSSIBLE_TRANSFER]",
                            "transfer : out std_logic;")
    architecture = architecture.replace(
      "[POSSIBLE_TRANSFER]",
      "transfer <= oehb_ready and join_valid;")
  else:
    entity = entity.replace("    [POSSIBLE_TRANSFER]\n", "")
    architecture = architecture.replace("  [POSSIBLE_TRANSFER]\n", "")

  return dependencies + entity + architecture

def _generate_addf_double_precision(name, export_transfer=False):
  join_name = f"{name}_join"
  oehb_name = f"{name}_oehb"
  buff_name = f"{name}_buff"
  ieee2nfloat_name = f"{name}_ieee2nfloat"
  nfloat2ieee_name = f"{name}_nfloat2ieee"
  floating_point_adder_name = f"{name}_floating_point_adder"
  dependencies = generate_join(join_name, {"size": 2}) + \
    generate_oehb(oehb_name, {"data_type": "!handshake.channel<i1>"}) + \
    generate_delay_buffer(buff_name, {"slots": _get_latency(is_double=True) - 1}) + \
    generate_input_ieee_64bit(ieee2nfloat_name) + \
    generate_output_ieee_64bit(nfloat2ieee_name) + \
    generate_floating_point_adder_64bit(floating_point_adder_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
  port (
    [POSSIBLE_TRANSFER]
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

  ieee2nfloat_lhs: entity work.{ieee2nfloat_name}(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.{ieee2nfloat_name}(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.{nfloat2ieee_name}(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.{floating_point_adder_name}(arch)
    port map (
        clk => clk,
        ce  => oehb_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );
end architecture;
"""

  if export_transfer:
    entity = entity.replace("[POSSIBLE_TRANSFER]",
                            "transfer : out std_logic;")
    architecture = architecture.replace(
      "[POSSIBLE_TRANSFER]",
      "transfer <= oehb_ready and join_valid;")
  else:
    entity = entity.replace("    [POSSIBLE_TRANSFER]\n", "")
    architecture = architecture.replace("  [POSSIBLE_TRANSFER]\n", "")

  return dependencies + entity + architecture

def _generate_addf_signal_manager(name, data_type, is_double):
  inner_name = f"{name}_inner"
  dependencies = _generate_addf(name, is_double, export_transfer=True)

  if "spec" in data_type.extra_signals:
    dependencies += _generate_ofifo(f"{name}_spec_ofifo", {
      "slots": _get_latency(is_double), # todo: correct?
      "data_type": "!handshake.channel<i1>" })

  extra_signal_logic = {
    "spec": ("""
    signal spec_tfifo_in : std_logic_vector(0 downto 0);
    signal spec_tfifo_out : std_logic_vector(0 downto 0);
  """, f"""
    spec_inner(0) <= lhs_spec or rhs_spec;
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

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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

  for signal_name in data_type.extra_signals:
    if signal_name not in extra_signal_logic:
      raise ValueError(f"Extra signal {signal_name} is not supported")

  architecture = f"""
architecture arch of {name} is
  signal transfer : std_logic;
  [EXTRA_SIGNAL_SIGNAL_DECLS]
begin

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
