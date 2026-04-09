from generators.support.signal_manager.tag import generate_entity_tag_operations, generate_inner_port_forwarding_tag_operations
from generators.support.signal_manager.utils.generation import generate_signal_wise_forwarding

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
  elif data_bitwidth == 0:
    return _generate_untagger_dataless(name, current_tag, tag_bitwidth)
  else:
    return _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth)


def _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth):

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of untagger
entity {name} is
port(
  clk        : in std_logic;
  rst        : in std_logic;
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
end entity;
"""

  architecture = f"""
-- Architecture of untagger
architecture arch of {name} is
begin
    outs_valid<= ins_valid;
    tagOut_valid<= ins_valid;
    ins_ready <= tagOut_ready and outs_ready;
    outs <= ins;
    tagOut <= ins_{current_tag};
end architecture;
"""

  return entity + architecture


def _generate_untagger_dataless(name, current_tag, tag_bitwidth):

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of untagger
entity {name} is
port(
  clk       : in std_logic;
  rst       : in std_logic;
  ins_valid : in std_logic;

  outs_ready : in std_logic; 
  outs_valid : out std_logic;

  ins_ready : out std_logic;

  tagOut : out std_logic_vector({tag_bitwidth}-1 downto 0);
  tagOut_valid : out  std_logic;
  tagOut_ready : in std_logic;

  ins_{current_tag} : in std_logic_vector({tag_bitwidth}-1 downto 0) 
);
end {name};
"""

  architecture = f"""
-- Architecture of untagger
architecture arch of {name} is
begin
    outs_valid<= ins_valid;
    tagOut_valid<= ins_valid;
    ins_ready <= tagOut_ready and outs_ready;
    tagOut <= ins_{current_tag};
end architecture;
"""

  return entity + architecture


def _generate_untagger_signal_manager(name, data_bitwidth, current_tag, tag_bitwidth, extra_signals):
  inner_name = f"{name}_inner"
  inner = (
      _generate_untagger_dataless(name, current_tag, tag_bitwidth)
      if data_bitwidth == 0
      else _generate_untagger(name, data_bitwidth, current_tag, tag_bitwidth)
  )

  in_ports = [{
      "name": "ins",
      "bitwidth": data_bitwidth,
      "extra_signals": extra_signals
  }, {
      "name": f"ins_{current_tag}",
      "bitwidth": tag_bitwidth,
      "extra_signals": {},
      "handshaked": False
  }]

  out_ports = [{
      "name": "outs",
      "bitwidth": data_bitwidth,
      "extra_signals": extra_signals
  }, {
      "name": "tagOut",
      "bitwidth": tag_bitwidth,
      "extra_signals": {}
  }]

  entity = generate_entity_tag_operations(name, in_ports, out_ports)


  # Assign all extra signals for each output port, based on forwarded_extra_signals.
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []

  in_channel_names = [p["name"] for p in in_ports]

  for signal_name in extra_signals:

    out_channel_names = [
        p["name"]
        for p in out_ports
        if signal_name in p.get("extra_signals", {})
    ]

    if out_channel_names:
      extra_signal_assignments.append(
          generate_signal_wise_forwarding(
              in_channel_names,
              out_channel_names,
              signal_name
          )
      )

  extra_signal_assignments_formatted = "\n".join(extra_signal_assignments)
  inner_port_forwarding = generate_inner_port_forwarding_tag_operations(
      in_ports + out_ports)

  architecture = f"""
-- Architecture of untagger signal manager
architecture arch of {name} is
begin
  -- Forward extra signals to output ports
  {extra_signal_assignments_formatted}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {inner_port_forwarding}
    );
end architecture;
"""

  return inner + entity + architecture