from generators.support.utils import VhdlScalarType, generate_extra_signal_ports_arrays, ExtraSignalMapping, generate_ins_concat_statements_dataless, generate_outs_concat_statements_dataless
from generators.support.join import generate_join
from generators.handshake.fork import _generate_fork

def generate_tagger(name, params):
  size = params["size"]
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["ins"])
  tag_bitwidth = VhdlScalarType(port_types["tagIn"]).bitwidth

  if data_type.has_extra_signals():
    return _generate_tagger_signal_manager(name, size, data_type, tag_bitwidth)
  else:
    return _generate_tagger(name, size, data_type, tag_bitwidth)

def _generate_tagger(name, size, data_type, tag_bitwidth):
  join_name = f"{name}_join"
  fork_name = f"{name}_fork"

  data_bitwidth = data_type.bitwidth

  dependencies = \
      generate_join(join_name, {
          "size": size+1
      }) + \
      _generate_fork(fork_name, size, 1) 

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity {name} is
  port(
    clk, rst      : in  std_logic;
    ins_valid : in std_logic_vector({size}-1 downto 0);

    outs_ready : in std_logic_vector({size} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);

    ins_ready : out std_logic_vector({size}-1 downto 0);

    ins   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    outs  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

    tagIn : in std_logic_vector({tag_bitwidth}-1 downto 0);
    tagIn_valid : in  std_logic;
    tagIn_ready : out std_logic;

    outs_tag : out data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""

  architecture = f"""
architecture arch of {name} is
signal join_valid : std_logic;
signal join_nReady : std_logic;
signal combined_valid : std_logic_vector({size} downto 0);
signal combined_ready : std_logic_vector({size} downto 0);
constant all_one : std_logic_vector({size}-1 downto 0) := (others => '1');

signal join_ins_ready : std_logic_vector({size} downto 0);

signal fork_ready: std_logic;
signal fork_useless_out : data_array({size} - 1 downto 0)(0 downto 0);

begin
    -- Combine tagIn_valid and ins_valid
    combined_valid <= tagIn_valid & ins_valid;

    j : entity work.{join_name}
                port map(   combined_valid,
                            join_nReady,
                            join_valid,
                            combined_ready);

    outs <= ins;

    -- Split combined_ready into ins_ready and tagIn_ready
    ins_ready   <= combined_ready({size}-1 downto 0);
    tagIn_ready <= combined_ready({size});

    tagging_process : process (freeTag_data)
    begin
      for I in 0 to {size} - 1 loop
        outs_tag(I)({tag_bitwidth}-1 downto 0) <= freeTag_data;
      end loop;
    end process;

    join_nReady <= fork_ready; 

    f : entity work.{fork_name}(arch)
            port map (
        --inputs
            clk => clk, 
            rst => rst,  
            ins => "1",
            ins_valid => join_valid,
            ins_ready => fork_ready, 
        --outputs
            outs => fork_useless_out,
            outs_valid => outs_valid,   
            outs_ready => outs_ready
            );

end architecture;
"""
  return dependencies + entity + architecture

def _generate_tagger_signal_manager(name, size, data_type, tag_bitwidth):
  inner_name = f"{name}_inner"

  data_bitwidth = data_type.bitwidth  

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_type in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  extra_signals_total_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_tagger(inner_name, size, data_type, tag_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity {name} is
  port(
    clk, rst      : in  std_logic;
    [EXTRA_SIGNAL_PORTS]
    ins_valid : in std_logic_vector({size}-1 downto 0);

    outs_ready : in std_logic_vector({size} - 1 downto 0);
    outs_valid : out std_logic_vector({size} - 1 downto 0);

    ins_ready : out std_logic_vector({size}-1 downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

    ins   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    outs  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

    tagIn : in std_logic_vector({tag_bitwidth}-1 downto 0);
    tagIn_valid : in  std_logic;
    tagIn_ready : out std_logic;

    outs_tag : out data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""
  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports_arrays([
      ("ins", "in", {size}),
      ("outs", "out", {size})
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
architecture arch of {name} is
  signal ins_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
begin

  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => ins_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready => ins_ready,
      ins => ins,
      outs => outs,
      tagIn => tagIn,
      tagIn_valid => tagIn_valid,
      tagIn_ready => tagIn_ready,
      outs_tag => outs_tag
    );

    outs_inner <= ins_inner;

end architecture;
"""
  ins_conversion = generate_ins_concat_statements_dataless(
      "ins", "ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless(
      "outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture