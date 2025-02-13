import ast

from generators.support.utils import VhdlScalarType
from generators.support.array import generate_2d_array
from generators.handshake.tfifo import generate_tfifo
from generators.handshake.cond_br import generate_cond_br
from generators.handshake.merge import generate_merge

def generate_spec_commit(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])

  # TODO: Support extra signals other than spec
  if data_type.is_channel():
    return _generate_spec_commit(name, data_type.bitwidth)
  else:
    return _generate_spec_commit_dataless(name)

def _generate_spec_commit(name, bitwidth):
  fifo_disc_name = f"{name}_fifo_disc"
  cond_br_name = f"{name}_cond_br"
  buff_name = f"{name}_buff"
  merge_name = f"{name}_merge"
  array_name = f"{name}_array"

  dependencies = \
    generate_tfifo(fifo_disc_name, {
      "size": 1,
      "port_types": str({
        "ins": "!handshake.channel<i1>",
        "outs": "!handshake.channel<i1>"
      })
    }) + \
    generate_cond_br(cond_br_name, {
      "port_types": str({
        "data": f"!handshake.channel<i{bitwidth}>",
        "condition": f"!handshake.channel<i{bitwidth}>",
        "trueOut": f"!handshake.channel<i{bitwidth}>",
        "falseOut": f"!handshake.channel<i{bitwidth}>"
      })
    }) + \
    generate_tfifo(buff_name, {
      "size": 1,
      "port_types": str({
        "ins": f"!handshake.channel<i{bitwidth}>",
        "outs": f"!handshake.channel<i{bitwidth}>"
      })
    }) + \
    generate_merge(merge_name, {
      "size": 2,
      "port_types": str({
        "ins_0": f"!handshake.channel<i{bitwidth}>",
        "ins_1": f"!handshake.channel<i{bitwidth}>",
        "outs": f"!handshake.channel<i{bitwidth}>"
      })
    }) + \
    generate_2d_array(array_name, 2, bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.{array_name}.all;

-- Entity of spec_commit
entity {name} is
  port (
    clk, rst : in  std_logic;
    -- inputs
    ins : in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of spec_commit
architecture arch of {name} is

signal fifo_disc_outs : std_logic_vector(0 downto 0);
signal fifo_disc_outs_valid : std_logic;
signal fifo_disc_outs_ready : std_logic;

signal branch_in_condition : std_logic_vector(0 downto 0);
signal branch_in_condition_ready : std_logic;
signal branch_in_trueOut : std_logic_vector({bitwidth} - 1 downto 0);
signal branch_in_trueOut_valid : std_logic;
signal branch_in_trueOut_ready : std_logic;
signal branch_in_falseOut : std_logic_vector({bitwidth} - 1 downto 0);
signal branch_in_falseOut_valid : std_logic;
signal branch_in_falseOut_ready : std_logic;

signal buff_outs : std_logic_vector({bitwidth} - 1 downto 0);
signal buff_outs_valid : std_logic;
signal buff_outs_ready : std_logic;

signal branch_disc_trueOut : std_logic_vector({bitwidth} - 1 downto 0);
signal branch_disc_trueOut_valid : std_logic;
signal branch_disc_trueOut_ready : std_logic;
signal branch_disc_falseOut : std_logic_vector({bitwidth} - 1 downto 0);
signal branch_disc_falseOut_valid : std_logic;
signal branch_disc_falseOut_ready : std_logic;

signal merge_ins : {array_name};
signal merge_ins_valid : std_logic_vector(1 downto 0);
signal merge_ins_ready : std_logic_vector(1 downto 0);

begin

-- Design taken directly from the Speculation 2019 paper

fifo_disc: entity work.{fifo_disc_name}(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => ctrl,
    ins_valid => ctrl_valid,
    ins_ready => ctrl_ready,
    outs => fifo_disc_outs,
    outs_valid => fifo_disc_outs_valid,
    outs_ready => fifo_disc_outs_ready
  );

branch_in_condition <= ins_spec;
branch_in: entity work.{cond_br_name}(arch)
  port map (
    clk => clk,
    rst => rst,
    data => ins,
    data_valid => ins_valid,
    data_ready => ins_ready,
    condition => branch_in_condition,
    condition_valid => '1', -- always valid
    condition_ready => branch_in_condition_ready,
    trueOut => branch_in_trueOut,
    trueOut_valid => branch_in_trueOut_valid,
    trueOut_ready => branch_in_trueOut_ready,
    falseOut => branch_in_falseOut,
    falseOut_valid => branch_in_falseOut_valid,
    falseOut_ready => branch_in_falseOut_ready
  );

buff: entity work.{buff_name}(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => branch_in_trueOut,
    ins_valid => branch_in_trueOut_valid,
    ins_ready => branch_in_trueOut_ready,
    outs => buff_outs,
    outs_valid => buff_outs_valid,
    outs_ready => buff_outs_ready
  );

branch_disc_trueOut_ready <= '1'; -- sink
branch_disc: entity work.{cond_br_name}(arch)
  port map (
    clk => clk,
    rst => rst,
    data => buff_outs,
    data_valid => buff_outs_valid,
    data_ready => buff_outs_ready,
    condition => fifo_disc_outs,
    condition_valid => fifo_disc_outs_valid,
    condition_ready => fifo_disc_outs_ready,
    trueOut => branch_disc_trueOut,
    trueOut_valid => branch_disc_trueOut_valid,
    trueOut_ready => branch_disc_trueOut_ready,
    falseOut => branch_disc_falseOut,
    falseOut_valid => branch_disc_falseOut_valid,
    falseOut_ready => branch_disc_falseOut_ready
  );

merge_ins <= (branch_disc_falseOut, branch_in_falseOut);
merge_ins_valid <= (branch_disc_falseOut_valid, branch_in_falseOut_valid);
branch_disc_falseOut_ready <= merge_ins_ready(1);
branch_in_falseOut_ready <= merge_ins_ready(0);

merge_out: entity work.{merge_name}(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => merge_ins,
    ins_valid => merge_ins_valid,
    ins_ready => merge_ins_ready,
    outs => outs,
    outs_valid => outs_valid,
    outs_ready => outs_ready
  );

end architecture;
"""

  return dependencies + entity + architecture

def _generate_spec_commit_dataless(name):
  inner_name = f"{name}_inner"

  dependencies = _generate_spec_commit(inner_name, 1)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_commit_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- inputs
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;
"""

  architecture = f"""
architecture arch of spec_commit_dataless_with_tag is
  signal ins_inner : std_logic_vector(0 downto 0);
  signal outs_inner : std_logic_vector(0 downto 0);
begin
  ins_inner(0) <= '0';
  spec_commit : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_spec => ins_spec,
      ctrl => ctrl,
      ctrl_valid => ctrl_valid,
      outs_ready => outs_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      ins_ready => ins_ready,
      ctrl_ready => ctrl_ready
    );
end architecture;
"""

  return dependencies + entity + architecture
