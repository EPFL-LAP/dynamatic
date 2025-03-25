def generate_trunci(name, params):
  input_bitwidth = params["input_bitwidth"]
  output_bitwidth = params["output_bitwidth"]

  return _generate_trunci(name, input_bitwidth, output_bitwidth)


def _generate_trunci(name, input_bitwidth, output_bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of trunci
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
-- Architecture of trunci
architecture arch of {name} is
begin
  outs       <= ins({output_bitwidth} - 1 downto 0);
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
end architecture;
"""

  return entity + architecture
