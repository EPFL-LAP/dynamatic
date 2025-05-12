from generators.support.signal_manager import generate_signal_manager
from generators.handshake.join import generate_join


def generate_demux(name, params):
  # Number of data output ports
  size = params["size"]

  data_bitwidth = params["data_bitwidth"]
  index_bitwidth = params["index_bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_demux_signal_manager(name, size, data_bitwidth, index_bitwidth, extra_signals)
  elif data_bitwidth == 0:
    return _generate_demux_dataless(name, size, index_bitwidth)
  else:
    return _generate_demux(name, size, data_bitwidth, index_bitwidth)


def _generate_demux_dataless(name, size, index_bitwidth):
  join_name = f"{name}_join"

  dependencies = generate_join(join_name, {"size": 2})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of demux_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} -1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channels
    outs_valid : out  std_logic_vector({size} - 1 downto 0);
    outs_ready : in std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of demux_dataless
architecture arch of {name} is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.{join_name}(arch)
    port map(
      clk          => clk,
      rst          => rst,
      -- input channels
      ins_valid(0) => ins_valid,
      ins_valid(1) => index_valid,
      ins_ready(0) => ins_ready,
      ins_ready(1) => index_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  process (index, branchInputs_valid)
    variable temp : std_logic_vector(8 - 1 downto 0);
  begin
    for i in 0 to {size}-1 loop
      outs_valid(i) <= '0';
    end loop;

    outs_valid(to_integer(unsigned(index))) <= branchInputs_valid;
  end process;

  branch_ready   <= outs_ready(to_integer(unsigned(index)));
end architecture;
"""

  return dependencies + entity + architecture


def _generate_demux(name, size, data_bitwidth, index_bitwidth):
  inner_name = f"{name}_inner"

  dependencies = _generate_demux_dataless(inner_name, size, index_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of demux
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    ins       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} -1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- data output channels
    outs       : out  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    outs_valid : out  std_logic_vector({size} - 1 downto 0);
    outs_ready : in   std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of demux
architecture arch of {name} is
begin
  control : entity work.{inner_name}
    port map(
      clk             => clk,
      rst             => rst,
      ins_valid      => ins_valid,
      ins_ready      => ins_ready,
      index       => index,
      index_valid => index_valid,
      index_ready => index_ready,
      outs_valid   => outs_valid,
      outs_ready   => outs_ready
    );

  process (ins)
  begin
    for i in 0 to {size}-1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;
"""

  return dependencies + entity + architecture


def _generate_demux_signal_manager(name, size, data_bitwidth, index_bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "ins",
          "bitwidth": data_bitwidth,
          "extra_signals": extra_signals
      }, {
          "name": "index",
          "bitwidth": index_bitwidth,
          "extra_signals": {}
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": data_bitwidth,
          "2d": True,
          "size": size,
          "extra_signals": extra_signals
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_demux_dataless(name, size, index_bitwidth) if data_bitwidth == 0
      else _generate_demux(name, size, data_bitwidth, index_bitwidth))
