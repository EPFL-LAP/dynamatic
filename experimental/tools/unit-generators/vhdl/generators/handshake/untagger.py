from generators.support.utils import VhdlScalarType, generate_extra_signal_ports_arrays, ExtraSignalMapping, generate_ins_concat_statements_dataless, generate_outs_concat_statements_dataless
from generators.support.join import generate_join

def generate_untagger(name, params):
  size = params["size"]
  port_types = params["port_types"]
  data_in_type = VhdlScalarType(port_types["ins"])
  data_out_type = VhdlScalarType(port_types["outs"])
  tag_bitwidth = VhdlScalarType(port_types["tagOut"]).bitwidth


    # Get extra signals from input data type
  input_extra_signals = data_in_type.extra_signals

  # Get extra signals from output data type
  output_extra_signals = data_out_type.extra_signals

  # Get the current tag that was removed by the tagger by viewing the difference of tags between the input and output data types
  unique_data_out_signals = {
    name: bitwidth
    for name, bitwidth in input_extra_signals.items()
    if name not in output_extra_signals
  }
  current_tag = next(iter(unique_data_out_signals))

  if data_out_type.has_extra_signals():
    return _generate_untagger_signal_manager(name, size, data_out_type, current_tag, tag_bitwidth)
  else:
    return _generate_untagger(name, size, data_out_type, current_tag, tag_bitwidth)

def _generate_untagger(name, size, data_out_type, current_tag, tag_bitwidth):
  join_name = f"{name}_join"

  data_bitwidth = data_out_type.bitwidth

  dependencies = \
      generate_join(join_name, {
          "size": size +1
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
  ins_valid : in std_logic_vector({size} - 1 downto 0);

  outs_ready : in std_logic_vector({size} - 1 downto 0); 
  outs_valid : out std_logic_vector({size} - 1 downto 0);

  ins_ready : out std_logic_vector({size} - 1 downto 0);

  ins   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
  outs  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

  tagOut : out std_logic_vector({tag_bitwidth}-1 downto 0);
  tagOut_valid : in  std_logic;
  tagOut_ready : out std_logic;

  ins_{current_tag} : in data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
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

begin
    -- Combine tagOut_valid and ins_valid
    combined_valid <= tagOut_valid & ins_valid;

    j : entity work.{join_name}(arch)
                port map(   combined_valid,
                            join_nReady, --outs_ready(0),
                            join_valid,
                            combined_ready);
    
    -- Split combined_ready into ins_ready and tagOut_ready
    ins_ready   <= combined_ready({size}-1 downto 0);
    tagOut_ready <= combined_ready({size});

    outs <= ins;

    tagOut <= ins_{current_tag}(0)({tag_bitwidth}-1 downto 0);  -- take the tag of any of the inputs; they are all guaranteed to be the same

    process(join_valid)
    begin
        if(join_valid = '1') then 
            outs_valid <= (others => '1');
        else
            outs_valid <= (others => '0');
        end if;
    end process;

    process (outs_ready)
        variable check : std_logic := '1';
    begin
        check := '1';
        for I in 0 to {size}-1 loop
            check := check and outs_ready(I);
        end loop;
        if(check = '1') then
            join_nReady <= '1';
        else 
            join_nReady <= '0';
        end if;
    end process;
end architecture;
"""
  return dependencies + entity + architecture


def _generate_untagger_signal_manager(name, size, data_out_type, current_tag, tag_bitwidth):
  inner_name = f"{name}_inner"

  data_bitwidth = data_out_type.bitwidth  

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_type in data_out_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  extra_signals_total_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_untagger(inner_name, size, data_out_type, tag_bitwidth) 

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
  ins_valid : in std_logic_vector({size} - 1 downto 0);

  outs_ready : in std_logic_vector({size} - 1 downto 0);  -- this signal and the one after include the extra output of the UNTAGGER that carries the freed up tag
  outs_valid : out std_logic_vector({size} -1 downto 0);

  ins_ready : out std_logic_vector({size} - 1 downto 0);

  ins   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
  outs  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

  tagOut : out std_logic_vector({tag_bitwidth}-1 downto 0);
  tagOut_valid : in  std_logic;
  tagOut_ready : out std_logic;

  ins_{current_tag} : in data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
);
end {name};
"""
  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports_arrays([
      ("ins", "in", size),
      ("outs", "out", size)
  ], data_out_type.extra_signals)
  entity = entity.replace("  [EXTRA_SIGNAL_PORTS]", extra_signal_ports)

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
      tagOut => tagOut,
      tagOut_valid => tagOut_valid,
      tagOut_ready => tagOut_ready,
      ins_{current_tag} => ins_{current_tag}
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