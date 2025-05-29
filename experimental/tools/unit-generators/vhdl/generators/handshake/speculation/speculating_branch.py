from generators.handshake.cond_br import generate_cond_br
from generators.support.utils import data
from generators.support.signal_manager.utils.entity import generate_entity


def generate_speculating_branch(name, params):
    data_bitwidth = params["data_bitwidth"]
    spec_tag_bitwidth = params["spec_tag_bitwidth"]
    extra_signals = params["extra_signals"]

    # Always contains spec signal
    if len(extra_signals) > 1:
        return _generate_speculating_branch_signal_manager(name, data_bitwidth, spec_tag_bitwidth, extra_signals)
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
    {data(f"data       : in  std_logic_vector({data_bitwidth} - 1 downto 0);", data_bitwidth)}
    data_valid : in  std_logic;
    data_spec : in std_logic_vector(0 downto 0);
    data_ready : out std_logic;
    -- spec_tag_data used for condition
    {data(f"spec_tag_data       : in  std_logic_vector({spec_tag_data_bitwidth} - 1 downto 0);", spec_tag_data_bitwidth)}
    spec_tag_data_valid : in  std_logic;
    spec_tag_data_spec : in std_logic_vector(0 downto 0);
    spec_tag_data_ready : out std_logic;
    -- true output channel
    {data(f"trueOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);", data_bitwidth)}
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    {data(f"falseOut       : out std_logic_vector({data_bitwidth} - 1 downto 0);", data_bitwidth)}
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
      {data("data => data,", data_bitwidth)}
      data_valid => data_valid,
      data_spec => data_spec,
      data_ready => data_ready,
      condition => cond_br_condition,
      condition_valid => spec_tag_data_valid,
      condition_spec => data_spec,
      condition_ready => spec_tag_data_ready,
      {data("trueOut => trueOut,", data_bitwidth)}
      trueOut_valid => trueOut_valid,
      trueOut_spec => open,
      trueOut_ready => trueOut_ready,
      {data("falseOut => falseOut,", data_bitwidth)}
      falseOut_valid => falseOut_valid,
      falseOut_spec => open,
      falseOut_ready => falseOut_ready
    );

end architecture;
"""

    return dependencies + entity + architecture


def _generate_speculating_branch_signal_manager(name, data_bitwidth, spec_tag_data_bitwidth, extra_signals):
    # Extra signals are discarded except for `spec`

    inner_name = f"{name}_inner"
    inner = _generate_speculating_branch(
        inner_name, data_bitwidth, spec_tag_data_bitwidth)

    entity = generate_entity(name, [{
        "name": "data",
        "bitwidth": data_bitwidth,
        "extra_signals": extra_signals
    }, {
        "name": "spec_tag_data",
        "bitwidth": spec_tag_data_bitwidth,
        "extra_signals": extra_signals
    }], [{
        "name": "trueOut",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }, {
        "name": "falseOut",
        "bitwidth": data_bitwidth,
        "extra_signals": {}
    }])

    architecture = f"""
-- Architecture of speculating_branch signal manager
architecture arch of {name} is
begin
  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {data("data => data,", data_bitwidth)}
      data_valid => data_valid,
      data_ready => data_ready,
      data_spec => data_spec,
      {data("spec_tag_data => spec_tag_data,", spec_tag_data_bitwidth)}
      spec_tag_data_valid => spec_tag_data_valid,
      spec_tag_data_ready => spec_tag_data_ready,
      spec_tag_data_spec => spec_tag_data_spec,
      {data("trueOut => trueOut_inner,", data_bitwidth)}
      trueOut_valid => trueOut_inner_valid,
      trueOut_ready => trueOut_inner_ready,
      {data("falseOut => falseOut_inner,", data_bitwidth)}
      falseOut_valid => falseOut_inner_valid,
      falseOut_ready => falseOut_inner_ready
    );
end architecture;
"""

    return inner + entity + architecture
