from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat, generate_slice
from generators.handshake.tehb import generate_tehb
from generators.handshake.ofifo import generate_ofifo


def generate_load(name, params):
    addr_bitwidth = params["addr_bitwidth"]
    data_bitwidth = params["data_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_load_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals)
    else:
        return _generate_load(name, data_bitwidth, addr_bitwidth)


def _generate_load(name, data_bitwidth, addr_bitwidth):
    addr_tehb_name = f"{name}_addr_tehb"
    data_tehb_name = f"{name}_data_tehb"

    dependencies = \
        generate_tehb(addr_tehb_name, {"bitwidth": addr_bitwidth}) + \
        generate_tehb(data_tehb_name, {"bitwidth": data_bitwidth})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of load
entity {name} is
  port (
    clk, rst : in std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic;
    -- data from interface channel
    dataFromMem       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    dataFromMem_valid : in  std_logic;
    dataFromMem_ready : out std_logic;
    -- data from memory channel
    dataOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of load
architecture arch of {name} is
begin
  addr_tehb : entity work.{addr_tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => addrIn,
      ins_valid => addrIn_valid,
      ins_ready => addrIn_ready,
      -- output channel
      outs       => addrOut,
      outs_valid => addrOut_valid,
      outs_ready => addrOut_ready
    );

  data_tehb : entity work.{data_tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => dataFromMem,
      ins_valid => dataFromMem_valid,
      ins_ready => dataFromMem_ready,
      -- output channel
      outs       => dataOut,
      outs_valid => dataOut_valid,
      outs_ready => dataOut_ready
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_load_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals):
    # Concatenate extra signals and store them in a dedicated FIFO

    # Get concatenation details for extra signals
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_total_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_load(inner_name, data_bitwidth, addr_bitwidth)

    # Generate ofifo to store extra signals for in-flight memory requests
    ofifo_name = f"{name}_ofifo"
    ofifo = generate_ofifo(ofifo_name, {
        "bitwidth": extra_signals_total_bitwidth,
        "num_slots": 1  # Assume LoadOp is connected to a memory controller
    })

    entity = generate_entity(name, [{
        "name": "addrIn",
        "bitwidth": addr_bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "dataFromMem",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }], [{
        "name": "addrOut",
        "bitwidth": addr_bitwidth,
        "extra_signals": {}
    }, {
        "name": "dataOut",
        "bitwidth": data_bitwidth,
        "extra_signals": extra_signals
    }])

    assignments = []

    # Concatenate addrIn extra signals to create signals_pre_buffer
    assignments.extend(generate_concat(
        "addrIn", 0, "signals_pre_buffer", concat_layout))

    # Slice signals_post_buffer to create dataOut data and extra signals
    assignments.extend(generate_slice(
        "signals_post_buffer", "dataOut", 0, concat_layout))

    architecture = f"""
-- Architecture of load signal manager
architecture arch of {name} is
  signal signals_pre_buffer, signals_post_buffer : std_logic_vector({concat_layout.total_bitwidth} - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
  -- Transfer signal assignments
  transfer_in <= addrIn_valid and addrIn_ready;
  transfer_out <= dataOut_valid and dataOut_ready;

  -- Concat/slice extra signals
  {"\n  ".join(assignments)}

  -- Buffer to store extra signals for in-flight memory requests
  -- LoadOp is assumed to be connected to a memory controller
  -- Use ofifo with latency 1 (MC latency)
  ofifo : entity work.{ofifo_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => signals_pre_buffer,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => signals_post_buffer,
      outs_valid => open,
      outs_ready => transfer_out
    );

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready,
      dataFromMem => dataFromMem,
      dataFromMem_valid => dataFromMem_valid,
      dataFromMem_ready => dataFromMem_ready,
      dataOut => dataOut,
      dataOut_valid => dataOut_valid,
      dataOut_ready => dataOut_ready
    );
end architecture;
"""

    return inner + ofifo + entity + architecture
