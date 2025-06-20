from generators.support.signal_manager import generate_buffered_signal_manager
from generators.handshake.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.oehb import generate_oehb


def generate_mulf(name, params):
    is_double = params["is_double"]
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_mulf_signal_manager(name, is_double, extra_signals)
    else:
        return _generate_mulf(name, is_double)


def _generate_mulf(name, is_double):
    if is_double:
        return _generate_mulf_double_precision(name)
    else:
        return _generate_mulf_single_precision(name)


def _get_latency(is_double):
    # doesn't depend on the bitwidth
    return 4


def _generate_mulf_single_precision(name):
    join_name = f"{name}_join"
    buff_name = f"{name}_buff"
    oehb_name = f"{name}_oehb"

    dependencies = generate_join(join_name, {"size": 2}) + \
        generate_delay_buffer(buff_name, {"slots": _get_latency(is_double=False) - 1}) + \
        generate_oehb(oehb_name, {"bitwidth": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mulf_single_precision
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
-- Architecture of mulf_single_precision
architecture arch of {name} is
  signal join_valid             : std_logic;
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

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.{oehb_name}(arch)
  port map(
    clk        => clk,
    rst        => rst,
    ins_valid  => buff_valid,
    outs_ready => result_ready,
    outs_valid => result_valid,
    ins_ready  => oehb_ready
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

  ip : entity work.FloatingPointMultiplier(arch)
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


def _generate_mulf_double_precision(name):
    join_name = f"{name}_join"
    oehb_name = f"{name}_oehb"
    buff_name = f"{name}_buff"

    dependencies = generate_join(join_name, {"size": 2}) + \
        generate_oehb(oehb_name, {"bitwidth": 0}) + \
        generate_delay_buffer(
        buff_name, {"slots": _get_latency(is_double=True) - 1})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mulf_double_precision
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
-- Architecture of mulf_double_precision
architecture arch of {name} is
  signal join_valid             : std_logic;
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

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
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

  ip : entity work.FPMult_64bit(arch)
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


def _generate_mulf_signal_manager(name, is_double, extra_signals):
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
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_mulf(name, is_double),
        _get_latency(is_double))
