from generators.support.signal_manager import generate_signal_manager

def generate_extract(name, params):
  bitwidth = params["bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_extract_signal_manager(name, bitwidth, extra_signals)
  if bitwidth == 0:
    return _generate_extract_dataless(name)
  else:
    return _generate_extract(name, bitwidth)
  
def _generate_extract(name, bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of extract
entity {name} is
  port(
    clk, rst      : in  std_logic;
    -- input channel
    ins   : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs  : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out  std_logic;
    outs_ready : in std_logic
  );
end {name};
"""
    
  architecture = f"""
-- Architecture of extract
architecture arch of {name} is
begin
  outs <= ins;
  outs_valid <= ins_valid;
  ins_ready <= outs_ready;
end architecture;
"""
  return entity + architecture

def _generate_extract_dataless(name):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of extract
entity {name} is
  port(
    clk, rst      : in  std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out  std_logic;
    outs_ready : in std_logic
  );
end {name};
"""
    
  architecture = f"""
-- Architecture of extract
architecture arch of {name} is
begin
  outs_valid <= ins_valid;
  ins_ready <= outs_ready;
end architecture;
"""
  return entity + architecture

def _generate_extract_signal_manager(name, bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": {}
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_extract_dataless(name) if bitwidth == 0
      else _generate_extract(name, bitwidth))