from generators.support.signal_manager import generate_signal_manager, get_concat_extra_signals_bitwidth
from generators.support.logic import generate_and_n

def generate_lazy_fork(name, params):
  # Number of output ports
  size = params["size"]

  bitwidth = params["bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_lazy_fork_signal_manager(name, size, bitwidth, extra_signals)
  elif bitwidth == 0:
    return _generate_lazy_fork_dataless(name, size)
  else:
    return _generate_lazy_fork(name, size, bitwidth)
  
def _generate_lazy_fork_dataless(name, size):
  and_n_module_name = f"{name}_and_n"

  dependencies = generate_and_n(and_n_module_name, {"size": size})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of lazy_fork_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""
  
  architecture = f"""
-- Architecture of lazy_fork_dataless
architecture arch of {name} is
  signal allnReady : std_logic;
begin
  genericAnd : entity work.{and_n_module_name} port map(outs_ready, allnReady);

  valids : process (ins_valid, outs_ready)
    variable tmp_ready : std_logic_vector({size} - 1 downto 0);
  begin
    for i in tmp_ready'range loop
      tmp_ready(i) := '1';
      for j in outs_ready'range loop
        if i /= j then
          tmp_ready(i) := (tmp_ready(i) and outs_ready(j));
        end if;
      end loop;
    end loop;
    for i in outs_valid'range loop
      outs_valid(i) <= ins_valid and tmp_ready(i);
    end loop;
  end process;

  ins_ready <= allnReady;
end architecture;
"""
  return dependencies + entity + architecture

def _generate_lazy_fork(name, size, bitwidth):
  inner_name = f"{name}_inner"

  dependencies = _generate_lazy_fork_dataless(inner_name, size)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of lazy_fork
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array ({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of lazy_fork
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

  process (ins)
  begin
    for i in 0 to {size} - 1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;
"""
  
  return dependencies + entity + architecture

def _generate_lazy_fork_signal_manager(name, size, bitwidth, extra_signals):
  extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
  return generate_signal_manager(name, {
      "type": "concat",
      "in_ports": [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals,
          "2d": True,
          "size": size
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_lazy_fork(name, size, bitwidth + extra_signals_bitwidth))
