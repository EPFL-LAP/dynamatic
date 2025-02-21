from generators.support.utils import VhdlScalarType
from generators.support.signal_manager.buffer import generate_buffer_like_signal_manager_full, generate_buffer_like_signal_manager_dataless_full
from generators.handshake.tehb import generate_tehb
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner

def generate_ofifo(name, params):
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["ins"])

  if data_type.has_extra_signals():
    if data_type.is_channel():
      return _generate_ofifo_signal_manager(name, params["size"], data_type)
    else:
      return _generate_ofifo_signal_manager_dataless(name, params["size"], data_type)
  elif data_type.is_channel():
    return _generate_ofifo(name, params["num_slots"], data_type.bitwidth)
  else:
    return _generate_ofifo_dataless(name, params["num_slots"])

def _generate_ofifo(name, size, bitwidth):
  tehb_name = f"{name}_tehb"
  fifo_name = f"{name}_fifo"

  dependencies = \
    generate_elastic_fifo_inner(fifo_name, {
      "size": size,
      "bitwidth": bitwidth
    }) + \
    generate_tehb(tehb_name, {
      "port_types": {
        "ins": f"!handshake.channel<i{bitwidth}>",
        "outs": f"!handshake.channel<i{bitwidth}>",
      }
    })

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
    generate_elastic_fifo_inner(fifo_name, {
      "size": size,
    }) + \
    generate_tehb(tehb_name, {
      "port_types": {
        "ins": "!handshake.control<>",
        "outs": "!handshake.control<>",
      }
    })

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

def _generate_ofifo_signal_manager(name, size, data_type):
  return generate_buffer_like_signal_manager_full(name, size, data_type, _generate_ofifo)

def _generate_ofifo_signal_manager_dataless(name, size, data_type):
  return generate_buffer_like_signal_manager_dataless_full(name, size, data_type, _generate_ofifo)
