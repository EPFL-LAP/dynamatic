from generators.support.signal_manager import generate_buffered_signal_manager
from generators.handshake.join import generate_join
from generators.handshake.oehb import generate_oehb


def generate_cmpf(name, params):
    is_double = params["is_double"]
    extra_signals = params["extra_signals"]
    predicate = params["predicate"]

    if extra_signals:
        return _generate_cmpf_signal_manager(name, is_double, predicate, extra_signals)
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
        generate_oehb(oehb_name, {"bitwidth": 0})

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


def _generate_cmpf_signal_manager(name, is_double, predicate, extra_signals):
    bitwidth = 64 if is_double else 32
    return generate_buffered_signal_manager(
        name,
        [{
            "name": "lhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "rhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "result",
            "bitwidth": 1,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_cmpf(name, is_double, predicate),
        _get_latency(is_double))
