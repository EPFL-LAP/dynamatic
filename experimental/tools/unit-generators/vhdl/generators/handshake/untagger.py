from generators.support.signal_manager import generate_signal_manager
from generators.handshake.join import generate_join

def generate_untagger(name, params):
  data_bitwidth = params["data_bitwidth"]
  tag_bitwidth = params["tag_bitwidth"]

  input_extra_signals = params.get("input_extra_signals", None)
  output_extra_signals = params.get("output_extra_signals", None)

  # Get the current tag that was removed by the tagger by viewing the difference of tags between the input and output data types
  unique_data_out_signals = {
    name: bitwidth
    for name, bitwidth in input_extra_signals.items()
    if name not in output_extra_signals
  }
  current_tag = next(iter(unique_data_out_signals))

  if output_extra_signals:
    return _generate_untagger_signal_manager(name, data_bitwidth, current_tag, tag_bitwidth, output_extra_signals)
  else:
    return _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth)

def _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth):
  join_name = f"{name}_join"

  dependencies = \
      generate_join(join_name, {
          "size": 1
      })
  
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity {name} is
port(
  clk, rst      : in  std_logic;
  ins_valid : in std_logic;

  outs_ready : in std_logic; 
  outs_valid : out std_logic;

  ins_ready : out std_logic;

  ins   : in  std_logic_vector({data_bitwidth} - 1 downto 0);
  outs  : out std_logic_vector({data_bitwidth} - 1 downto 0);

  tagOut : out std_logic_vector({tag_bitwidth}-1 downto 0);
  tagOut_valid : out  std_logic;
  tagOut_ready : in std_logic;

  ins_{current_tag} : in std_logic_vector({tag_bitwidth}-1 downto 0) 
);
end {name};
"""
  
  architecture = f"""
architecture arch of {name} is
begin
    outs_valid<= '1';
    tagOut_valid<= '1';
    ins_ready <= '1';
    outs <= ins;
    tagOut <= ins_{current_tag};

end architecture;
"""
  
  return dependencies + entity + architecture

def _generate_untagger_signal_manager(name, data_bitwidth, current_tag, tag_bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "ins",
          "bitwidth": data_bitwidth,
          "extra_signals": extra_signals
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": data_bitwidth,
          "extra_signals": extra_signals
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth))