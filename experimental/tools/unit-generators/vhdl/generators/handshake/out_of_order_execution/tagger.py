from generators.support.signal_manager import generate_entity_tag_operations, _forward_extra_signals, generate_inner_port_forwarding_tag_operations
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
                port map(   combined_valid,
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
                port map(   combined_valid,
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
  inner_name = f"{name}_inner"
  inner = (
      _generate_tagger_dataless(name, current_tag, tag_bitwidth)
      if data_bitwidth == 0
      else _generate_tagger(name, data_bitwidth, current_tag, tag_bitwidth)
  )

  in_ports = [{
      "name": "ins",
      "bitwidth": data_bitwidth,
      "extra_signals": extra_signals
  }, {
      "name": "tagIn",
      "bitwidth": tag_bitwidth,
      "extra_signals": {}
  }]

  out_ports = [{
      "name": "outs",
      "bitwidth": data_bitwidth,
      "extra_signals": extra_signals
  }, {
      "name": f"outs_{current_tag}",
      "bitwidth": tag_bitwidth,
      "extra_signals": {},
      "handshaked": False
  }]

  entity = generate_entity_tag_operations(name, in_ports, out_ports)

  forwarded_extra_signals = _forward_extra_signals(
      extra_signals, in_ports)

  # Assign all extra signals for each output port, based on forwarded_extra_signals.
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []
  for out_port in out_ports:
    port_name = out_port["name"]
    port_2d = out_port.get("2d", False)

    if not port_2d:
      # Assign all extra signals to this output port
      for signal_name in out_port["extra_signals"]:
        extra_signal_assignments.append(
            f"  {port_name}_{signal_name} <= {forwarded_extra_signals[signal_name]};")
    else:
      port_size = out_port["size"]
      for signal_name in out_port["extra_signals"]:
        for i in range(port_size):
          extra_signal_assignments.append(
              f"  {port_name}_{i}_{signal_name} <= {forwarded_extra_signals[signal_name]};")
  extra_signal_assignments_formatted = "\n".join(
      extra_signal_assignments).lstrip()

  inner_port_forwarding = generate_inner_port_forwarding_tag_operations(
      in_ports + out_ports)

  architecture = f"""
-- Architecture of tagger signal manager
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
