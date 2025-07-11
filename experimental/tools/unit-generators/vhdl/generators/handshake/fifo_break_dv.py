from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner


def generate_fifo_break_dv(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_fifo_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_fifo_break_dv_dataless(name, num_slots)
    else:
        return _generate_fifo_break_dv(name, num_slots, bitwidth)


def _generate_fifo_break_dv(name, size, bitwidth):
    fifo_name = f"{name}_fifo"

    dependencies = \
        generate_elastic_fifo_inner(fifo_name, {
            "size": size,
            "bitwidth": bitwidth
        })
    
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_dv
entity {name} is
  port (
    clk : in std_logic;
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
-- Architecture of fifo_break_dv
architecture arch of {name} is
  signal fifo_valid, fifo_ready     : std_logic;
  signal fifo_dataOut : std_logic_vector({bitwidth} - 1 downto 0);
begin

  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      --outputs
      outs       => outs,
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );

end architecture;
"""

    return dependencies + entity + architecture


def _generate_fifo_break_dv_dataless(name, size):
    fifo_name = f"{name}_fifo"

    dependencies = \
        generate_elastic_fifo_inner(fifo_name, {"size": size})
    entity = f"""

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_dv_dataless
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
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
-- Architecture of fifo_break_dv_dataless
architecture arch of {name} is
begin
  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      --outputs
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );

  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
"""

    return dependencies + entity + architecture


def _generate_fifo_break_dv_signal_manager(name, size, bitwidth, extra_signals):
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
        lambda name: _generate_fifo_break_dv(name, size, bitwidth + extra_signals_bitwidth))
