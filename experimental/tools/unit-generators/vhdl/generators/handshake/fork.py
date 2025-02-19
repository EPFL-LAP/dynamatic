import ast

from generators.support.utils import VhdlScalarType
from generators.support.logic import generate_or_n
from generators.support.eager_fork_register_block import generate_eager_fork_register_block

def generate_fork(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])
  size = int(params["size"])

  if data_type.is_channel():
    return _generate_fork(name, size, data_type.bitwidth)
  else:
    return _generate_fork_dataless(name, size)

def _generate_fork_dataless(name, size):
  or_n_name = f"{name}_or_n"
  regblock_name = f"{name}_regblock"

  dependencies = \
    generate_or_n(or_n_name, {"size": size}) + \
    generate_eager_fork_register_block(regblock_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fork_dataless
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
-- Architecture of fork_dataless
architecture arch of {name} is
  signal blockStopArray : std_logic_vector({size} - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.{or_n_name}
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in {size} - 1 downto 0 generate
    regblock : entity work.{regblock_name}(arch)
      port map(
        -- inputs
        clk          => clk,
        rst          => rst,
        ins_valid    => ins_valid,
        outs_ready   => outs_ready(i),
        backpressure => backpressure,
        -- outputs
        outs_valid => outs_valid(i),
        blockStop  => blockStopArray(i)
      );
  end generate;

end architecture;
"""

  return dependencies + entity + architecture

def _generate_fork(name, size, bitwidth):
  inner_name = f"{name}_inner"

  dependencies = _generate_fork_dataless(inner_name, size)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of fork
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);
    outs_ready : in  std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of fork
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
