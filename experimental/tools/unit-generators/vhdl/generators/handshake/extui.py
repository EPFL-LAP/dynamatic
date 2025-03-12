from generators.support.utils import VhdlScalarType

def generate_extui(name, params):
  port_types = params["port_types"]
  input_bitwidth = VhdlScalarType(port_types["ins"]).bitwidth
  output_bitwidth = VhdlScalarType(port_types["outs"]).bitwidth

  return _generate_extui(name, input_bitwidth, output_bitwidth)

def _generate_extui(name, input_bitwidth, output_bitwidth):

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is 
port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({input_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({output_bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""
  architecture = f"""
-- Architecture of extui
architecture arch of {name} is

begin
  outs({output_bitwidth} - 1 downto {input_bitwidth}) <= ({output_bitwidth} - {input_bitwidth} - 1 downto 0 => '0');
  outs({input_bitwidth} - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;
"""
  return entity + architecture