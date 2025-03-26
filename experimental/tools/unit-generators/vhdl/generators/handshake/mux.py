from generators.support.signal_manager import generate_signal_manager, get_concat_extra_signals_bitwidth
from generators.handshake.tehb import generate_tehb


def generate_mux(name, params):
  # Number of data input ports
  size = params["size"]

  data_bitwidth = params["data_bitwidth"]
  index_bitwidth = params["index_bitwidth"]

  # List of extra signals for each data input port
  # Each element is a dictionary where key: extra signal name, value: bitwidth
  # e.g., [{"tag0": 8, "spec": 1}, {"tag0": 8}]
  input_extra_signals_list = params["input_extra_signals_list"]
  # e.g., {"tag0": 8, "spec": 1}
  output_extra_signals = params["output_extra_signals"]
  index_extra_signals = params["index_extra_signals"]

  # List of indices of input ports that have spec bit
  # e.g., [0]
  spec_inputs = params["spec_inputs"]

  if output_extra_signals:
    return _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, index_extra_signals, spec_inputs)
  elif data_bitwidth == 0:
    return _generate_mux_dataless(name, size, index_bitwidth)
  else:
    return _generate_mux(name, size, index_bitwidth, data_bitwidth)


def _generate_mux(name, size, index_bitwidth, data_bitwidth):
  tehb_name = f"{name}_tehb"

  dependencies = generate_tehb(tehb_name, {"bitwidth": data_bitwidth})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

-- Entity of mux
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  data_array({size} - 1 downto 0)({data_bitwidth} - 1 downto 0);
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of mux
architecture arch of {name} is
  signal tehb_ins                       : std_logic_vector({data_bitwidth} - 1 downto 0);
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
begin
  process (ins, ins_valid, outs_ready, index, index_valid, tehb_ins_ready)
    variable selectedData                   : std_logic_vector({data_bitwidth} - 1 downto 0);
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in {size} - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready    <= (not index_valid) or (selectedData_valid and tehb_ins_ready);
    tehb_ins       <= selectedData;
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => tehb_ins,
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  return dependencies + entity + architecture


def _generate_mux_dataless(name, size, index_bitwidth):
  tehb_name = f"{name}_tehb"

  dependencies = generate_tehb(tehb_name, {"bitwidth": 0})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

-- Entity of mux_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins_valid : in  std_logic_vector({size} - 1 downto 0);
    ins_ready : out std_logic_vector({size} - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector({index_bitwidth} - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of mux_dataless
architecture arch of {name} is
  signal tehb_ins_valid, tehb_ins_ready : std_logic;
begin
  process (ins_valid, outs_ready, index, index_valid, tehb_ins_ready)
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData_valid := '0';

    for i in {size} - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;

      if indexEqual and index_valid and ins_valid(i) then
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and tehb_ins_ready) or (not ins_valid(i));
    end loop;

    index_ready    <= (not index_valid) or (selectedData_valid and tehb_ins_ready);
    tehb_ins_valid <= selectedData_valid;
  end process;

  tehb : entity work.{tehb_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins_valid => tehb_ins_valid,
      ins_ready => tehb_ins_ready,
      -- output channel
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  return dependencies + entity + architecture


def _generate_mux_signal_manager(name, size, index_bitwidth, data_bitwidth, input_extra_signals_list, output_extra_signals, index_extra_signals, spec_inputs):
  extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
      output_extra_signals)
  return generate_signal_manager(name, {
      "type": "bbmerge",
      "in_ports": [{
          "name": "ins",
          "bitwidth": data_bitwidth,
          "2d": True,
          "size": size,
          "extra_signals_list": input_extra_signals_list
      }, {
          "name": "index",
          "bitwidth": index_bitwidth,
          # TODO: Extra signals for index port are not tested
          "extra_signals": index_extra_signals
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": data_bitwidth,
          "extra_signals": output_extra_signals
      }],
      "size": size,
      "data_in_name": "ins",
      "index_name": "index",
      "index_dir": "in",
      "index_extra_signals": index_extra_signals,
      "out_extra_signals": output_extra_signals,
      "spec_inputs": spec_inputs
  }, lambda name: _generate_mux(name, size, index_bitwidth, extra_signals_bitwidth + data_bitwidth))
