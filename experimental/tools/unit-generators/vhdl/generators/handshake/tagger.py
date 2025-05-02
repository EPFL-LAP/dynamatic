from generators.support.signal_manager import generate_signal_manager
from generators.handshake.join import generate_join


def generate_tagger(name, params):
  data_bitwidth = params["data_bitwidth"]
  tag_bitwidth = params["tag_bitwidth"]

  input_extra_signals = params.get("input_extra_signals", None)
  output_extra_signals = params.get("output_extra_signals", None)

  # Get the current tag that was added by the tagger by viewing the difference of tags between the input and output data types
  unique_data_out_signals = {
      name: bitwidth
      for name, bitwidth in output_extra_signals.items()
      if name not in input_extra_signals
  }
  current_tag = next(iter(unique_data_out_signals))

  if input_extra_signals:
    return _generate_tagger_signal_manager(name, data_bitwidth, current_tag, tag_bitwidth, input_extra_signals)
  elif data_bitwidth == 0:
    return _generate_tagger_dataless(name, current_tag, tag_bitwidth)
  else:
    return _generate_tagger(name, data_bitwidth, current_tag, tag_bitwidth)


def _generate_tagger(name, data_bitwidth, current_tag, tag_bitwidth):
  join_name = f"{name}_join"

  dependencies = \
      generate_join(join_name, {
          "size": 2
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of tagger
entity {name} is
  port(
    clk, rst      : in  std_logic;
    ins_valid : in std_logic;
    
    outs_ready : in std_logic; 
    outs_valid : out std_logic;

    ins_ready : out std_logic;

    ins   : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    outs  : out std_logic_vector({data_bitwidth} - 1 downto 0);

    tagIn : in std_logic_vector({tag_bitwidth}-1 downto 0);
    tagIn_valid : in  std_logic;
    tagIn_ready : out std_logic;

    outs_{current_tag} : out std_logic_vector({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""

  architecture = f"""
-- Architecture of tagger
architecture arch of {name} is
  signal combined_valid : std_logic_vector(1 downto 0);
  signal combined_ready : std_logic_vector(1 downto 0);
begin
    -- Combine tagIn_valid and ins_valid
    combined_valid <= tagIn_valid & ins_valid;

    j : entity work.{join_name}
                port map(   clk,
                            rst,
                            combined_valid,
                            outs_ready,
                            outs_valid,
                            combined_ready);

    outs <= ins;

    -- Split combined_ready into ins_ready and tagIn_ready
    ins_ready   <= combined_ready(0);
    tagIn_ready <= combined_ready(1);

    outs_{current_tag} <= tagIn;

end architecture;
"""

  return dependencies + entity + architecture

def _generate_tagger_dataless(name, current_tag, tag_bitwidth):
  join_name = f"{name}_join"

  dependencies = \
      generate_join(join_name, {
          "size": 2
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of tagger
entity {name} is
  port(
    clk, rst      : in  std_logic;
    ins_valid : in std_logic;
    
    outs_ready : in std_logic; 
    outs_valid : out std_logic;

    ins_ready : out std_logic;

    tagIn : in std_logic_vector({tag_bitwidth}-1 downto 0);
    tagIn_valid : in  std_logic;
    tagIn_ready : out std_logic;

    outs_{current_tag} : out std_logic_vector({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""

  architecture = f"""
-- Architecture of tagger
architecture arch of {name} is
  signal combined_valid : std_logic_vector(1 downto 0);
  signal combined_ready : std_logic_vector(1 downto 0);
begin
    -- Combine tagIn_valid and ins_valid
    combined_valid <= tagIn_valid & ins_valid;

    j : entity work.{join_name}
                port map(   clk,
                            rst,
                            combined_valid,
                            outs_ready,
                            outs_valid,
                            combined_ready);

    -- Split combined_ready into ins_ready and tagIn_ready
    ins_ready   <= combined_ready(0);
    tagIn_ready <= combined_ready(1);

    outs_{current_tag} <= tagIn;

end architecture;
"""

  return dependencies + entity + architecture


def _generate_tagger_signal_manager(name, data_bitwidth, current_tag, tag_bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "ins",
          "bitwidth": data_bitwidth,
          "extra_signals": extra_signals
      }, {
          "name": "tagIn",
          "bitwidth": tag_bitwidth,
          "extra_signals": {}
      },
      ],
      "out_ports": [{
          "name": "outs",
          "bitwidth": data_bitwidth,
          "extra_signals": extra_signals
      },
          {
          "name": f"outs_{current_tag}",
          "bitwidth": tag_bitwidth,
          "extra_signals": {},
          "handshaked": False
      },
      ],
      "extra_signals": extra_signals
  }, lambda name: _generate_tagger_dataless(name, current_tag, tag_bitwidth) if data_bitwidth == 0
      else _generate_tagger(name, data_bitwidth, current_tag, tag_bitwidth))


