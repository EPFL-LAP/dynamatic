from generators.support.utils import VhdlScalarType

# todo: move to somewhere else (like utils.py)
header = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
"""

# todo: move to somewhere else (like utils.py)
def generate_extra_signal_ports(ports, extra_signals):
  return "    -- extra signal ports\n" + "\n".join([
    "\n".join([
      f"    {port}_{name} : {inout} std_logic_vector({bitwidth - 1} downto 0);"
      for name, bitwidth in extra_signals.items()
    ])
    for port, inout in ports
  ])

def generate_cond_br(name, params):
  data_type = VhdlScalarType(params["data_type"])

  if data_type.has_extra_signals():
    return _generate_cond_br_signal_manager(name, data_type)
  elif data_type.is_channel():
    return _generate_cond_br_dataless(name)
  else:
    return _generate_cond_br(name, data_type.bitwidth)

def _generate_cond_br_dataless(name):
  # todo: generate_join is not implemented
  dependencies = generate_join(f"{name}_join", {size: 2})

  entity = f"""
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
  """

  architecture = f"""
architecture arch of {name} is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.{name}_join(arch)
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;
  """

  return header + dependencies + entity + architecture

def _generate_cond_br(name, bitwidth):
  dependencies = _generate_cond_br_dataless(f"{name}_dataless")

  entity = f"""
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector({bitwidth} - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
  """

  architecture = f"""
architecture arch of {name} is
begin
  control : entity work.{name}_dataless
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;
  """

  return header + dependencies + entity + architecture

def _generate_cond_br_signal_manager(name, data_type):
  dependencies = ""
  if data_type.is_channel():
    dependencies = _generate_cond_br(f"{name}_inner", data_type.bitwidth)
  else:
    dependencies = _generate_cond_br_dataless(f"{name}_inner")

  entity = f"""
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
[POSSIBLE_DATA_PORTS]
[EXTRA_SIGNAL_PORTS]
    data_valid : in std_logic;
    data_ready : out std_logic;
    condition : in std_logic_vector(0 downto 0);
    condition_valid : in std_logic;
    condition_ready : out std_logic
    trueOut_valid : out std_logic;
    trueOut_ready : in std_logic;
    falseOut_valid : out std_logic;
    falseOut_ready : in std_logic
  );
end entity;
"""

  # Add data ports if the data type is a channel
  if data_type.is_channel():
    entity = entity.replace("[POSSIBLE_DATA_PORTS]\n", f"""
    -- data ports
    data : in std_logic_vector({data_type.bitwidth - 1} downto 0);
    trueOut : out std_logic_vector({data_type.bitwidth - 1} downto 0);
    falseOut : out std_logic_vector({data_type.bitwidth - 1} downto 0);
""")
  else:
    entity = entity.replace("[POSSIBLE_DATA_PORTS]\n", "")

  # Add extra signal ports
  entity = entity.replace("[EXTRA_SIGNAL_PORTS]\n",
    generate_extra_signal_ports([
      ("data", "in"), ("condition", "in"),
      ("trueOut", "out"), ("falseOut", "out")
    ], data_type.extra_signals))

  # todo: can be reusable among various unit generators
  extra_signal_logics = {
    "spec": """
  trueOut_spec <= data_spec or condition_spec;
  falseOut_spec <= data_spec or condition_spec;
""" # todo: generate_normal_spec_logics(["trueOut", "falseOut"], ["data", "condition"])
  }

  architecture = f"""
architecture arch of {name} is
begin

  -- list of logics for supported extra signals
[EXTRA_SIGNAL_LOGICS]

  inner : entity work.{name}_inner(arch)
    port map(
      clk => clk,
      rst => rst,
[POSSIBLE_DATA_PORTS]
      data_valid => data_valid,
      data_ready => data_ready,
      condition => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid => trueOut_valid,
      trueOut_ready => trueOut_ready,
      falseOut_valid => falseOut_valid,
      falseOut_ready => falseOut_ready
    );
"""

  if data_type.is_channel():
    architecture = architecture.replace("[POSSIBLE_DATA_PORTS]\n", f"""
      -- data ports
      data => data,
      trueOut => trueOut,
      falseOut => falseOut,
""")
  else:
    architecture = architecture.replace("[POSSIBLE_DATA_PORTS]\n", "")

  architecture.replace("[EXTRA_SIGNAL_LOGICS]", "\n".join([
    extra_signal_logics[name] for name in data_type.extra_signals
  ]))

  return header + dependencies + entity + architecture
