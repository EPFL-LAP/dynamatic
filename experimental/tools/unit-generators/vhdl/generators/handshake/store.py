from generators.support.signal_manager.utils.entity import generate_entity


def generate_store(name, params):
    data_bitwidth = params["data_bitwidth"]
    addr_bitwidth = params["addr_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_store_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals)
    else:
        return _generate_store(name, data_bitwidth, addr_bitwidth)


def _generate_store(name, data_bitwidth, addr_bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of store
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data from circuit channel
    dataIn       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    dataIn_valid : in  std_logic;
    dataIn_ready : out std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- data to interface channel
    dataToMem       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    dataToMem_valid : out std_logic;
    dataToMem_ready : in  std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector({addr_bitwidth} - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of store
architecture arch of {name} is
begin
  -- data
  dataToMem       <= dataIn;
  dataToMem_valid <= dataIn_valid;
  dataIn_ready    <= dataToMem_ready;
  -- addr
  addrOut         <= addrIn;
  addrOut_valid   <= addrIn_valid;
  addrIn_ready    <= addrOut_ready;
end architecture;
"""

    return entity + architecture


def _generate_store_signal_manager(name, data_bitwidth, addr_bitwidth, extra_signals):
    # Discard extra signals

    inner_name = f"{name}_inner"
    inner = _generate_store(inner_name, data_bitwidth, addr_bitwidth)

    entity = generate_entity(name, [{
        "name": "dataIn",
        "bitwidth": data_bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "addrIn",
        "bitwidth": addr_bitwidth,
        "extra_signals": extra_signals
    }], [{
        "name": "dataToMem",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }, {
        "name": "addrOut",
        "bitwidth": addr_bitwidth,
        "extra_signals": {}
    }])

    architecture = f"""
-- Architecture of store signal manager
architecture arch of {name} is
begin
  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      dataIn => dataIn,
      dataIn_valid => dataIn_valid,
      dataIn_ready => dataIn_ready,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready,
      dataToMem => dataToMem,
      dataToMem_valid => dataToMem_valid,
      dataToMem_ready => dataToMem_ready,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready
    );
end architecture;
"""

    return inner + entity + architecture
