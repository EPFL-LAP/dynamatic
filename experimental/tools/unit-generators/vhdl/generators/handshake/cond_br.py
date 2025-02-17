import ast

from generators.support.utils import VhdlScalarType
from generators.support.join import generate_join

# todo: move to somewhere else (like utils.py)
header = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
"""

def generate_cond_br(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["data"])

  if data_type.is_channel():
    return _generate_cond_br(name, data_type.bitwidth)
  else:
    return _generate_cond_br_dataless(name)

def _generate_cond_br_dataless(name):
  # todo: generate_join is not implemented
  dependencies = generate_join(f"{name}_join", 2)

  entity = f"""
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
architecture arch of {name} is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.{name}_join(arch)
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;
"""

  return dependencies + header + entity + architecture

def _generate_cond_br(name, bitwidth):
  dependencies = _generate_cond_br_dataless(f"{name}_dataless")

  entity = f"""
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector({bitwidth} - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
architecture arch of {name} is
begin
  control : entity work.{name}_dataless
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;
"""

  return dependencies + header + entity + architecture
