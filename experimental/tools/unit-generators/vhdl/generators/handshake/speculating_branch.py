from generators.handshake.cond_br import generate_cond_br


def generate_speculating_branch(name, params):
  data_bitwidth = params["data_bitwidth"]
  spec_tag_bitwidth = params["spec_tag_bitwidth"]

  return _generate_speculating_branch(name, data_bitwidth, spec_tag_bitwidth)


def _generate_speculating_branch(name, data_bitwidth, spec_tag_data_bitwidth):
  inner_name = f"{name}_inner"

  dependencies = generate_cond_br(inner_name, {
      "bitwidth": data_bitwidth,
      "extra_signals": {"spec": 1}
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
