import ast

from generators.support.utils import VhdlScalarType
from generators.handshake.cond_br import generate_cond_br
from generators.handshake.tfifo import generate_tfifo

def generate_speculating_branch(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_bitwidth = VhdlScalarType(port_types["data"]).bitwidth
  spec_tag_data_bitwidth = VhdlScalarType(port_types["spec_tag_data"]).bitwidth

  return _generate_speculating_branch(name, data_bitwidth, spec_tag_data_bitwidth)

def _generate_speculating_branch_inner(name, data_bitwidth, spec_tag_data_bitwidth):
  inner_name = f"{name}_inner"

  dependencies = generate_cond_br(inner_name, {
    "port_types": str({
      "data": f"!handshake.channel<i{data_bitwidth}, [spec: i1]>",
      "condition": "!handshake.channel<i1, [spec: i1]>",
      "trueOut": f"!handshake.channel<i{data_bitwidth}, [spec: i1]>",
      "falseOut": f"!handshake.channel<i{data_bitwidth}, [spec: i1]>"
    })
  })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of speculating_branch
entity {name} is
  port(
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    data_valid : in  std_logic;
    data_spec : in std_logic_vector(0 downto 0);
    data_ready : out std_logic;
    -- spec_tag_data used for condition
    spec_tag_data       : in  std_logic_vector({spec_tag_data_bitwidth} - 1 downto 0);
    spec_tag_data_valid : in  std_logic;
    spec_tag_data_spec : in std_logic_vector(0 downto 0);
    spec_tag_data_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of speculating_branch
architecture arch of {name} is
  signal cond_br_condition : std_logic_vector(0 downto 0);
begin

  cond_br_condition <= spec_tag_data_spec;
  cond_br : entity work.{inner_name}(arch)
    port map (
      clk => clk,
      rst => rst,
      data => data,
      data_valid => data_valid,
      data_spec => data_spec,
      data_ready => data_ready,
      condition => cond_br_condition,
      condition_valid => spec_tag_data_valid,
      condition_spec => data_spec,
      condition_ready => spec_tag_data_ready,
      trueOut => trueOut,
      trueOut_valid => trueOut_valid,
      trueOut_spec => open,
      trueOut_ready => trueOut_ready,
      falseOut => falseOut,
      falseOut_valid => falseOut_valid,
      falseOut_spec => open,
      falseOut_ready => falseOut_ready
    );

end architecture;
"""

  return dependencies + entity + architecture

def _generate_speculating_branch(name, data_bitwidth, spec_tag_data_bitwidth):
  inner_name = f"{name}_inner"
  tfifo_data_name = f"{name}_tfifo_data"
  tfifo_spec_tag_name = f"{name}_tfifo_spec_tag"

  dependencies = \
    _generate_speculating_branch_inner(inner_name, data_bitwidth, spec_tag_data_bitwidth) + \
    generate_tfifo(tfifo_data_name, {
      "num_slots": "32",
      "port_types": str({
        "ins": f"!handshake.channel<i{data_bitwidth}, [spec: i1]>",
        "outs": f"!handshake.channel<i{data_bitwidth}, [spec: i1]>",
      })
    }) + \
    generate_tfifo(tfifo_spec_tag_name, {
      "num_slots": "32",
      "port_types": str({
        "ins": f"!handshake.channel<i{spec_tag_data_bitwidth}, [spec: i1]>",
        "outs": f"!handshake.channel<i{spec_tag_data_bitwidth}, [spec: i1]>",
      })
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of speculating_branch wrapper
entity {name} is
  port(
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector({data_bitwidth} - 1 downto 0);
    data_valid : in  std_logic;
    data_spec : in std_logic_vector(0 downto 0);
    data_ready : out std_logic;
    -- spec_tag_data used for condition
    spec_tag_data       : in  std_logic_vector({spec_tag_data_bitwidth} - 1 downto 0);
    spec_tag_data_valid : in  std_logic;
    spec_tag_data_spec : in std_logic_vector(0 downto 0);
    spec_tag_data_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of speculating_branch wrapper
architecture arch of {name} is
  signal data_inner : std_logic_vector({data_bitwidth} - 1 downto 0);
  signal data_valid_inner : std_logic;
  signal data_spec_inner : std_logic_vector(0 downto 0);
  signal data_ready_inner : std_logic;
  signal spec_tag_data_inner : std_logic_vector({spec_tag_data_bitwidth} - 1 downto 0);
  signal spec_tag_data_valid_inner : std_logic;
  signal spec_tag_data_spec_inner : std_logic_vector(0 downto 0);
  signal spec_tag_data_ready_inner : std_logic;
begin

  data_buf : entity work.{tfifo_data_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => data,
      ins_valid => data_valid,
      ins_spec => data_spec,
      ins_ready => data_ready,
      outs => data_inner,
      outs_valid => data_valid_inner,
      outs_spec => data_spec_inner,
      outs_ready => data_ready_inner
    );
  spec_tag_buf : entity work.{tfifo_spec_tag_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => spec_tag_data,
      ins_valid => spec_tag_data_valid,
      ins_spec => spec_tag_data_spec,
      ins_ready => spec_tag_data_ready,
      outs => spec_tag_data_inner,
      outs_valid => spec_tag_data_valid_inner,
      outs_spec => spec_tag_data_spec_inner,
      outs_ready => spec_tag_data_ready_inner
    );

  speculating_branch : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      data => data_inner,
      data_valid => data_valid_inner,
      data_spec => data_spec_inner,
      data_ready => data_ready_inner,
      spec_tag_data => spec_tag_data_inner,
      spec_tag_data_valid => spec_tag_data_valid_inner,
      spec_tag_data_spec => spec_tag_data_spec_inner,
      spec_tag_data_ready => spec_tag_data_ready_inner,
      trueOut => trueOut,
      trueOut_valid => trueOut_valid,
      trueOut_ready => trueOut_ready,
      falseOut => falseOut,
      falseOut_valid => falseOut_valid,
      falseOut_ready => falseOut_ready
    );
end architecture;
"""

  return dependencies + entity + architecture
