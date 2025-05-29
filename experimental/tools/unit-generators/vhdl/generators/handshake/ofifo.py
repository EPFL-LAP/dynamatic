from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.handshake.tehb import generate_tehb
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner


def generate_ofifo(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_ofifo_signal_manager(name, num_slots, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_ofifo_dataless(name, num_slots)
    else:
        return _generate_ofifo(name, num_slots, bitwidth)


def _generate_ofifo(name, size, bitwidth):
    tehb_name = f"{name}_tehb"
    fifo_name = f"{name}_fifo"

    dependencies = \
        generate_elastic_fifo_inner(fifo_name, {
            "size": size,
            "bitwidth": bitwidth
        }) + \
        generate_tehb(tehb_name, {"bitwidth": bitwidth})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo
entity {name} is
  port (
    clk, rst : in std_logic;
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
-- Architecture of ofifo
architecture arch of {name} is
  signal tehb_valid, tehb_ready     : std_logic;
  signal fifo_valid, fifo_ready     : std_logic;
  signal tehb_dataOut, fifo_dataOut : std_logic_vector({bitwidth} - 1 downto 0);
begin
  tehb : entity work.{tehb_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs       => tehb_dataOut,
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins        => tehb_dataOut,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs       => fifo_dataOut,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs       <= fifo_dataOut;
  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
"""

    return dependencies + entity + architecture


def _generate_ofifo_dataless(name, size):
    tehb_name = f"{name}_tehb"
    fifo_name = f"{name}_fifo"

    dependencies = \
        generate_elastic_fifo_inner(fifo_name, {"size": size}) + \
        generate_tehb(tehb_name, {"bitwidth": 0})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of ofifo_dataless
architecture arch of {name} is
  signal tehb_valid, tehb_ready : std_logic;
  signal fifo_valid, fifo_ready : std_logic;
begin
  tehb : entity work.{tehb_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
"""

    return dependencies + entity + architecture


def _generate_ofifo_signal_manager(name, size, bitwidth, extra_signals):
    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
    return generate_concat_signal_manager(
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
        lambda name: _generate_ofifo(name, size, bitwidth + extra_signals_bitwidth))
