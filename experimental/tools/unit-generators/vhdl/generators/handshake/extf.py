from generators.support.signal_manager import generate_default_signal_manager


def generate_extf(name, params):
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_extf_signal_manager(name, extra_signals)
    else:
        return _generate_extf(name)

def _generate_single_to_double(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

-- Entity of single_to_double
entity {name} is
  port (
    ins  : in std_logic_vector(32 -1 downto 0);
    outs : out std_logic_vector(64 - 1 downto 0)
  );
end entity;
"""
    architecture = f"""

-- Architecture of single_to_double
architecture arch of {name} is
  signal float_value : float32;
  signal float_extended : float64;
begin
  float_value <= to_float(ins);
  float_extended <= to_float64(float_value);
  outs <= to_std_logic_vector(float_extended);
end architecture;

"""
    return entity + architecture




def _generate_extf(name):
    single_to_double_name = f"{name}_single_to_double"

    dependencies = _generate_single_to_double(single_to_double_name)
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extf
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(32 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(64 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

"""

    architecture = f"""
-- Architecture of extf
architecture arch of {name} is
begin
  converter: entity work.{single_to_double_name}(arch)
    port map (
      ins => ins,
      outs => outs
  );
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

"""

    return dependencies + entity + architecture


def _generate_extf_signal_manager(name, extra_signals):
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
        lambda name: _generate_extf(name))