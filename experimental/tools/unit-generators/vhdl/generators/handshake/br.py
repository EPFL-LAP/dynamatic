from generators.support.signal_manager import generate_default_signal_manager

from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.join import generate_join


def generate_br(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_br_signal_manager(name, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_br_dataless(name)
    else:
        return _generate_br(name, bitwidth)


def _generate_br_dataless(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of br_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- input channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of br_dataless
architecture arch of {name} is
begin
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_br(name, bitwidth):
    inner_name = f"{name}_inner"

    dependencies = _generate_br_dataless(inner_name)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of br
entity {name} is
  port (
    clk : in  std_logic;
    rst : in std_logic;
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
-- Architecture of br
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

  outs <= ins;
end architecture;

"""

    return dependencies + entity + architecture


def _generate_br_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name:
            (_generate_br_dataless(name) if bitwidth == 0
             else _generate_br(name, bitwidth)))
