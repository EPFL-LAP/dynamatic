from generators.support.signal_manager import generate_default_signal_manager


def generate_extui(name, params):
    input_bitwidth = params["input_bitwidth"]
    output_bitwidth = params["output_bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_extui_signal_manager(name, input_bitwidth, output_bitwidth, extra_signals)
    else:
        return _generate_extui(name, input_bitwidth, output_bitwidth)


def _generate_extui(name, input_bitwidth, output_bitwidth):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extui
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


def _generate_extui_signal_manager(name, input_bitwidth, output_bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": input_bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": output_bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_extui(name, input_bitwidth, output_bitwidth))
