from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner
from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_tfifo(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_tfifo_signal_manager(name, num_slots, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_tfifo_dataless(name, num_slots)
    else:
        return _generate_tfifo(name, num_slots, bitwidth)


def _generate_tfifo(name, size, bitwidth):
    fifo_inner_name = f"{name}_fifo"
    dependencies = \
        generate_elastic_fifo_inner(fifo_inner_name, {
            "size": size,
            "bitwidth": bitwidth,
        })

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo
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
-- Architecture of tfifo
architecture arch of {name} is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
  signal fifo_in, fifo_out        : std_logic_vector({bitwidth} - 1 downto 0);
begin

  process (mux_sel, fifo_out, ins) is
  begin
    if (mux_sel = '1') then
      outs <= fifo_out;
    else
      outs <= ins;
    end if;
  end process;

  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;

  fifo_nready <= outs_ready;
  fifo_in     <= ins;

  fifo : entity work.{fifo_inner_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => fifo_in,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs       => fifo_out,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_tfifo_dataless(name, size):
    fifo_inner_name = f"{name}_fifo"
    dependencies = generate_elastic_fifo_inner(fifo_inner_name, {"size": size})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo_dataless
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
-- Architecture of tfifo_dataless
architecture arch of {name} is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
begin
  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;
  fifo_nready <= outs_ready;

  fifo : entity work.{fifo_inner_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_tfifo_signal_manager(name, size, bitwidth, extra_signals):
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
        lambda name: _generate_tfifo(name, size, bitwidth + extra_signals_bitwidth))
