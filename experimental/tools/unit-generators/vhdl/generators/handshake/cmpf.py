from generators.support.utils import VhdlScalarType, generate_extra_signal_ports
from generators.support.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.support.oehb import generate_oehb
from generators.support.ofifo import generate_ofifo


def generate_cmpf(name, params):
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["lhs"])
  predicate = params["predicate"]

  if data_type.bitwidth == 32:
    is_double = False
  elif data_type.bitwidth == 64:
    is_double = True
  else:
    raise ValueError(f"Unsupported bitwidth {data_type.bitwidth}")

  if data_type.has_extra_signals():
    return _generate_cmpf_signal_manager(name, data_type, is_double, predicate)
  else:
    return _generate_cmpf(name, is_double, predicate)


_expression_from_predicate = {
    "oeq": "not unordered and XeqY",
    "ogt": "not unordered and XgtY",
    "oge": "not unordered and XgeY",
    "olt": "not unordered and XltY",
    "ole": "not unordered and XleY",
    "one": "not unordered and not XeqY",
    "ueq": "unordered or XeqY",
    "ugt": "unordered or XgtY",
    "uge": "unordered or XgeY",
    "ult": "unordered or XltY",
    "ule": "unordered or XleY",
    "une": "unordered or not XeqY",
    "uno": "unordered"
}


def _generate_cmpf(name, is_double, predicate):
  inner_name = f"{name}_inner"
  bitwidth = 64 if is_double else 32
  if is_double:
    dependencies = _generate_cmpf_double_precision(inner_name)
  else:
    dependencies = _generate_cmpf_single_precision(inner_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of cmpf
architecture arch of {name} is
  signal unordered : std_logic;
  signal XltY : std_logic;
  signal XeqY : std_logic;
  signal XgtY : std_logic;
  signal XleY : std_logic;
  signal XgeY : std_logic;
begin
  operator : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      rhs => rhs,
      rhs_valid => rhs_valid,
      result_ready => result_ready,
      unordered => unordered,
      XltY => XltY,
      XeqY => XeqY,
      XgtY => XgtY,
      XleY => XleY,
      XgeY => XgeY,
      result_valid => result_valid,
      lhs_ready => lhs_ready,
      rhs_ready => rhs_ready
    );

  result(0) <= {_expression_from_predicate[predicate]};
end architecture;
"""

  return dependencies + entity + architecture


def _get_latency(is_double):
  return 1  # todo


def _generate_cmpf_single_precision(name):
  join_name = f"{name}_join"

  dependencies = generate_join(join_name, {"size": 2})

  entity = f"""


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf_single_precision
entity {name} is
  port(
    -- inputs
    clk: in std_logic;
    rst: in std_logic;
    lhs: in std_logic_vector(32 - 1 downto 0);
    lhs_valid: in std_logic;
    rhs: in std_logic_vector(32 - 1 downto 0);
    rhs_valid: in std_logic;
    result_ready: in std_logic;
    -- outputs
    unordered: out std_logic;
    XltY: out std_logic;
    XeqY: out std_logic;
    XgtY: out std_logic;
    XleY: out std_logic;
    XgeY: out std_logic;
    result_valid: out std_logic;
    lhs_ready: out std_logic;
    rhs_ready: out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of cmpf_single_precision
architecture arch of {name} is
  signal ip_lhs: std_logic_vector(32 + 1 downto 0);
  signal ip_rhs: std_logic_vector(32 + 1 downto 0);
begin
  join_inputs: entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0)=> lhs_valid,
      ins_valid(1)=> rhs_valid,
      outs_ready=> result_ready,
      -- outputs
      outs_valid=> result_valid,
      ins_ready(0)=> lhs_ready,
      ins_ready(1)=> rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> lhs,
        --output
        R=> ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
    port map(
        --input
        X=> rhs,
        --output
        R=> ip_rhs
    );
  operator: entity work.FPComparator_32bit(arch)
  port map (clk=> clk,
        ce=> '1',
        X=> ip_lhs,
        Y=> ip_rhs,
        unordered=> unordered,
        XltY=> XltY,
        XeqY=> XeqY,
        XgtY=> XgtY,
        XleY=> XleY,
        XgeY=> XgeY);
end architecture;
"""

  return dependencies + entity + architecture


def _generate_cmpf_double_precision(name):
  join_name = f"{name}_join"
  oehb_name = f"{name}_oehb"

  dependencies = generate_join(join_name, {"size": 2}) + \
      generate_oehb(oehb_name, {"data_type": "!handshake.control<>"})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf_double_precision
entity {name} is
  port(
    -- inputs
    clk: in std_logic;
    rst: in std_logic;
    lhs: in std_logic_vector(64 - 1 downto 0);
    lhs_valid: in std_logic;
    rhs: in std_logic_vector(64 - 1 downto 0);
    rhs_valid: in std_logic;
    result_ready: in std_logic;
    -- outputs
    result: out std_logic_vector(64 - 1 downto 0);
    result_valid: out std_logic;
    lhs_ready: out std_logic;
    rhs_ready: out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of cmpf_double_precision
architecture arch of {name} is
  signal join_valid: std_logic;
	signal buff_valid, oehb_valid, oehb_ready : std_logic;
	signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
  signal ip_lhs : std_logic_vector(64 + 1 downto 0);
  signal ip_rhs : std_logic_vector(64 + 1 downto 0);
begin

 oehb : entity work.{oehb_name}(arch)
  port map(
    clk        => clk,
    rst        => rst,
    ins_valid  => buff_valid,
    outs_ready => result_ready,
    outs_valid => result_valid,
    ins_ready  => oehb_ready
  );
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => buff_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_64bit(arch)
    port map (
        --input
        X => lhs,
        --output
        R => ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_64bit(arch)
    port map (
        --input
        X => rhs,
        --output
        R => ip_rhs
    );
  operator : entity work.FPComparator_64bit(arch)
  port map (clk => clk,
        ce => oehb_ready,
        X => ip_lhs,
        Y => ip_rhs,
        unordered => unordered,
        XltY => XltY,
        XeqY => XeqY,
        XgtY => XgtY, 
        XleY => XleY, 
        XgeY => XgeY);
end architecture;
"""

  return dependencies + entity + architecture


def _generate_cmpf_signal_manager(name, data_type, is_double, predicate):
  inner_name = f"{name}_inner"
  bitwidth = 64 if is_double else 32

  dependencies = _generate_cmpf(inner_name, is_double, predicate)

  if "spec" in data_type.extra_signals:
    dependencies += generate_ofifo(f"{name}_spec_ofifo", {
        "num_slots": _get_latency(is_double),  # todo: correct?
        "port_types": {
            "ins": "!handshake.channel<i1>",
            "outs": "!handshake.channel<i1>"
        }
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
        ins_valid => transfer_in,
        ins_ready => open,
        outs => spec_tfifo_out,
        outs_valid => open,
        outs_ready => transfer_out
      );
    result_spec <= spec_tfifo_out;
""")
  }

  for signal_name in data_type.extra_signals:
    if signal_name not in extra_signal_logic:
      raise ValueError(f"Extra signal {signal_name} is not supported")

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpf signal manager
entity {name} is
  port (
    [EXTRA_SIGNAL_PORTS]
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
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
-- Architecture of cmpf signal manager
architecture arch of {name} is
  signal transfer_in, transfer_out : std_logic;
  [EXTRA_SIGNAL_SIGNAL_DECLS]
begin
  transfer_in <= lhs_valid and lhs_ready;
  transfer_out <= result_valid and result_ready;

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
