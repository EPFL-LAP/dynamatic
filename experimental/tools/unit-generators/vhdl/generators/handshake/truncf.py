from generators.support.signal_manager import generate_default_signal_manager


def generate_truncf(name, params):
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_truncf_signal_manager(name, extra_signals)
    else:
        return _generate_truncf(name)

def _generate_double_to_single(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

entity {name} is
  port (
    ins  : in std_logic_vector(63 downto 0);
    outs : out std_logic_vector(31 downto 0)
  );
end entity;

"""
    architecture = f"""
architecture arch of {name} is
  signal float_value : float64;
  signal float_truncated : float32;

begin
  float_value <= to_float(ins, 11, 52);
  float_truncated <= to_float32(float_value);
  outs <= to_std_logic_vector(float_truncated);
end architecture;

"""
    return entity + architecture




def _generate_truncf(name):
    double_to_single_name = f"{name}_double_to_single"

    dependencies = _generate_double_to_single(double_to_single_name)
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of truncf
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(64 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

"""

    architecture = f"""
-- Architecture of truncf
architecture arch of {name} is
begin
  converter: entity work.{double_to_single_name}(arch)
    port map (
      ins => ins,
      outs => outs
  );
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

"""

    return dependencies + entity + architecture


def _generate_truncf_signal_manager(name, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": 64,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": 32,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_truncf(name))
