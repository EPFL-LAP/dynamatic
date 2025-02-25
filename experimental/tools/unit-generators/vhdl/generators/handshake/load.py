from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements_dataless, generate_outs_concat_statements_dataless
from generators.handshake.tehb import generate_tehb
from generators.handshake.tfifo import generate_tfifo


def generate_load(name, params):
  port_types = params["port_types"]

  # Ports communicating with the elastic circuit have the complete and same extra signals
  data_type = VhdlScalarType(port_types["dataOut"])
  addr_type = VhdlScalarType(port_types["addrIn"])

  if data_type.has_extra_signals():
    return _generate_load_signal_manager(name, data_type, addr_type)
  else:
    return _generate_load(name, data_type.bitwidth, addr_type.bitwidth)


def _generate_load(name, data_bitwidth, addr_bitwidth):
  addr_tehb_name = f"{name}_addr_tehb"
  data_tehb_name = f"{name}_data_tehb"

  dependencies = \
      generate_tehb(addr_tehb_name, {
          "port_types": {
              "ins": f"!handshake.channel<i{addr_bitwidth}>",
              "outs": f"!handshake.channel<i{addr_bitwidth}>"
          }
      }) + \
      generate_tehb(data_tehb_name, {
          "port_types": {
              "ins": f"!handshake.channel<i{data_bitwidth}>",
              "outs": f"!handshake.channel<i{data_bitwidth}>"
          }
      })

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


def _generate_load_signal_manager(name, data_type, addr_type):
  inner_name = f"{name}_inner"
  tfifo_name = f"{name}_tfifo"

  data_bitwidth = data_type.bitwidth
  addr_bitwidth = addr_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_type in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  extra_signals_total_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_load(inner_name, data_bitwidth, addr_bitwidth) + \
      generate_tfifo(tfifo_name, {
          "port_types": {
              "ins": f"!handshake.channel<i{extra_signals_total_bitwidth}>",
              "outs": f"!handshake.channel<i{extra_signals_total_bitwidth}>"
          },
          "num_slots": 32  # todo
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of load signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
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

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
      ("addrIn", "in"),
      ("dataOut", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of load signal manager
architecture arch of {name} is
  signal addrIn_ready_inner : std_logic;
  signal tfifo_ready : std_logic;
  signal tfifo_n_ready : std_logic;
  signal tfifo_ins_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
  signal tfifo_outs_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
begin
  addrIn_ready <= addrIn_ready_inner and tfifo_ready;
  tfifo_n_ready <= dataOut_valid and dataOut_ready;

  [EXTRA_SIGNAL_LOGIC]

  tfifo : entity work.{tfifo_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => tfifo_ins_inner,
      ins_valid => addrIn_valid and addrIn_ready_inner,
      ins_ready => tfifo_ready,
      outs => tfifo_outs_inner,
      outs_valid => open,
      outs_ready => tfifo_n_ready
    );

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready_inner,
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

  ins_conversion = generate_ins_concat_statements_dataless(
      "addrIn", "tfifo_ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless(
      "dataOut", "tfifo_outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture
