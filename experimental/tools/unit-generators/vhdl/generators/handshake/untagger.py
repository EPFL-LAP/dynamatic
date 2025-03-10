from generators.support.utils import VhdlScalarType, generate_extra_signal_ports_arrays, generate_ins_concat_statements_dataless, generate_outs_concat_statements_dataless
from generators.support.join import generate_join

def generate_untagger(name, params):
  size = params["size"]
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["DATA_TYPE"])
  tag_bitwidth = VhdlScalarType(port_types["TAG_TYPE"]).bitwidth

  if data_type.has_extra_signals():
    return _generate_untagger_signal_manager(name, data_type, tag_bitwidth)
  else:
    return _generate_untagger(name, data_type, tag_bitwidth)

def _generate_untagger(name, size, data_type, tag_bitwidth):
  join_name = f"{name}_join"

  data_bitwidth = data_type.bitwidth

  dependencies = \
      generate_join(join_name, {
          "size": size
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
  pValidArray : in std_logic_vector({size} - 1 downto 0);

  nReadyArray : in std_logic_vector({size} downto 0);  -- this signal and the one after include the extra output of the UNTAGGER that carries the freed up tag
  validArray : out std_logic_vector({size} downto 0);

  readyArray : out std_logic_vector({size} - 1 downto 0);

  dataInArray   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
  dataOutArray  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

  freeTag_data : out std_logic_vector({tag_bitwidth}-1 downto 0);

  tagInArray : in data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
);
end {name};
"""

  architecture = f"""
signal join_valid : std_logic;
signal join_nReady : std_logic;
constant all_one : std_logic_vector({size}-1 downto 0) := (others => '1');

signal join_readyArray : std_logic_vector({size} downto 0);

begin
    
    j : entity work.{join_name}(arch)
                port map(   pValidArray,
                            join_nReady, --nReadyArray(0),
                            join_valid,
                            readyArray);

    dataOutArray <= dataInArray;

    freeTag_data <= tagInArray(0)({tag_bitwidth} downto 0);  -- take the tag of any of the inputs; they are all guaranteed to be the same

    process(join_valid)
    begin
        if(join_valid = '1') then 
            validArray <= (others => '1');
        else
            validArray <= (others => '0');
        end if;
    end process;

    process (nReadyArray)
        variable check : std_logic := '1';
    begin
        check := '1';
        for I in 0 to {size} loop
            check := check and nReadyArray(I);
        end loop;
        if(check = '1') then
            join_nReady <= '1';
        else 
            join_nReady <= '0';
        end if;
    end process;
architecture arch of {name} is
end architecture;
"""
  return dependencies + entity + architecture


def _generate_untagger_signal_manager(name, size, data_type, tag_bitwidth):
  inner_name = f"{name}_inner"

  data_bitwidth = data_type.bitwidth  

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_type in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  extra_signals_total_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_untagger(inner_name, size, data_type, tag_bitwidth) 
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
  pValidArray : in std_logic_vector({size} - 1 downto 0);

  nReadyArray : in std_logic_vector({size} downto 0);  -- this signal and the one after include the extra output of the UNTAGGER that carries the freed up tag
  validArray : out std_logic_vector({size} downto 0);

  readyArray : out std_logic_vector({size} - 1 downto 0);

  dataInArray   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
  dataOutArray  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

  freeTag_data : out std_logic_vector({tag_bitwidth}-1 downto 0);

  tagInArray : in data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
);
end {name};
"""
  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports_arrays([
      ("dataInArray", "in", {size}),
      ("dataOutArray", "out", {size})
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
      pValidArray => pValidArray,
      nReadyArray => nReadyArray,
      validArray => validArray,
      readyArray => readyArray,
      dataInArray => dataInArray,
      dataOutArray => dataOutArray,
      freeTag_data => freeTag_data,
      tagInArray => tagInArray
    );

    outs_inner <= ins_inner;

end architecture;
"""
  ins_conversion = generate_ins_concat_statements_dataless(
      "dataInArray", "ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless(
      "dataOutArray", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture