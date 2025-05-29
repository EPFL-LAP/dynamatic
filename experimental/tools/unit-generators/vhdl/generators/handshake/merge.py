from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.handshake.merge_notehb import generate_merge_notehb
from generators.handshake.tehb import generate_tehb


def generate_merge(name, params):
    # Number of intput ports
    size = params["size"]
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_merge_signal_manager(name, size, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_merge_dataless(name, size)
    else:
        return _generate_merge(name, size, bitwidth)


def _generate_merge_dataless(name, size):
    inner_name = f"{name}_inner"
    tehb_name = f"{name}_tehb"

    dependencies = generate_merge_notehb(inner_name, {"size": size}) + \
        generate_tehb(tehb_name, {"bitwidth": 0, "size": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of merge_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of merge_dataless
architecture arch of {name} is
  signal tehb_pvalid : std_logic;
  signal tehb_ready  : std_logic;
begin
  merge_ins : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_merge(name, size, bitwidth):
    inner_name = f"{name}_inner"
    tehb_name = f"{name}_tehb"

    dependencies = \
        generate_merge_notehb(inner_name, {
            "size": size,
            "bitwidth": bitwidth,
        }) + \
        generate_tehb(tehb_name, {"bitwidth": bitwidth})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of merge
architecture arch of {name} is
  signal tehb_data_in : std_logic_vector({bitwidth} - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin

  merge_ins : entity work.{inner_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs       => tehb_data_in,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_merge_signal_manager(name, size, bitwidth, extra_signals):
    # Haven't tested this function yet
    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
    return generate_concat_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals,
            "size": size
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_merge(name, size, bitwidth + extra_signals_bitwidth))
