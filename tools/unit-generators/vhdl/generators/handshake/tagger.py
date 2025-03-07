from generators.support.utils import VhdlScalarType, generate_extra_signal_ports
from generators.handshake.fork.py import _generate_fork

def generate_tagger(name, params):
  dataOperands = port_types["dataOperands"]
  size = len(dataOperands)
  data_type = VhdlScalarType(dataOperands[0])
  tag_bitwidth = VhdlScalarType(port_types["tagOperand"]).bitwidth

  if data_type.has_extra_signals():
    return _generate_tagger_signal_manager(name, data_type, addr_type)
  else:
    return _generate_tagger(name, data_type.bitwidth, tag_bitwidth)

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
    pValidArray : in std_logic_vector({size} downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

    nReadyArray : in std_logic_vector({size} - 1 downto 0);
    validArray : out std_logic_vector({size} - 1 downto 0);

    readyArray : out std_logic_vector({size} downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

    dataInArray   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    dataOutArray  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

    freeTag_data : in std_logic_vector({tag_bitwidth}-1 downto 0);

    tagOut : out data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""
  # Add extra signal ports: possibly tags coming from outer graph
  extra_signal_ports = generate_extra_signal_ports([
      ("addrIn", "in"),
      ("dataOut", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
architecture arch of {name} is
signal join_valid : std_logic;
signal join_nReady : std_logic;
constant all_one : std_logic_vector({size}-1 downto 0) := (others => '1');

signal join_readyArray : std_logic_vector({size} downto 0);

signal fork_ready: STD_LOGIC_VECTOR ({size} - 1 downto 0);
signal fork_useless_out : data_array({size} - 1 downto 0)(0 downto 0);

begin
    
    j : entity work.{join_name}
                port map(   pValidArray,
                            join_nReady,
                            join_valid,
                            readyArray);

    dataOutArray <= dataInArray;

    tagging_process : process (freeTag_data)
    begin
      for I in 0 to {size} - 1 loop
        tagOut(I)({TAG_SIZE} downto 0) <= freeTag_data;
      end loop;
    end process;

    join_nReady <= fork_ready(0); 

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
            outs_valid => validArray,   
            outs_ready => fork_ready
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

  dependencies = _generate_tagger(inner_name, size, data_type, tag_bitwidth) + \
      generate_tfifo(tfifo_name, {
          "port_types": {
              "ins": f"!handshake.channel<i{extra_signals_total_bitwidth}>",
              "outs": f"!handshake.channel<i{extra_signals_total_bitwidth}>"
          },
          "num_slots": 32  # todo
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
    [EXTRA_SIGNAL_PORTS]
    pValidArray : in std_logic_vector({size} downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

    nReadyArray : in std_logic_vector({size} - 1 downto 0);
    validArray : out std_logic_vector({size} - 1 downto 0);

    readyArray : out std_logic_vector({size} downto 0); -- doesnot have a -1 because it includes the pValid of the freeTag_data input too

    dataInArray   : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    dataOutArray  : out data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);

    freeTag_data : in std_logic_vector({tag_bitwidth}-1 downto 0);

    tagOut : out data_array ({size} - 1 downto 0)({tag_bitwidth}-1 downto 0) 
  );
end {name};
"""
  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
      ("dataInArray", "in"),
      ("dataOutArray", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

    architecture = f"""
architecture arch of {name} is
  signal nReadyArray : std_logic_vector({size} - 1 downto 0);
  signal tfifo_ready : std_logic;
  signal tfifo_n_ready : std_logic;
  signal tfifo_ins_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
  signal tfifo_outs_inner : std_logic_vector({extra_signals_total_bitwidth} - 1 downto 0);
begin
  addrIn_ready <= nReadyArray and tfifo_ready;
  tfifo_n_ready <= dataOut_valid and dataOut_ready;

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
      tagOut => tagOut
    );

end architecture;
"""