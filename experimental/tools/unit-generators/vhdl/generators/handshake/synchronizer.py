from generators.handshake.join import generate_join
from generators.handshake.fork import generate_fork
from generators.support.utils import data


def generate_synchronizer(name, params):
  size = params["size"]

  bitwidths = []
  for i in range(size):
      bitwidths.append(params[f"bitwidth_{i}"])

  input_ports = ""
  for i, bitwidth in enumerate(bitwidths):
      input_ports += input_port(i, bitwidth)
  input_ports = input_ports.lstrip()

  output_ports = ""
  for i, bitwidth in enumerate(bitwidths):
      output_ports += output_port(i, bitwidth)
  output_ports = output_ports.lstrip().rpartition(";")[0].rstrip()

  join_name = f"{name}_join"
  fork_name = f"{name}_fork"

  dependencies = generate_join(join_name, {"size": size}) + \
                generate_fork(fork_name, {"size": size, "bitwidth": 0})


  entity = f"""
  library ieee;
  use ieee.std_logic_1164.all;

  -- Entity of synchronizer
  entity {name} is
    port (
      {input_ports}
      {output_ports}
    );
  end entity;
  """

  valid_instance_assignments = ""
  for i in range(size):
      valid_instance_assignments += f"    ins_valid({i}) => ins{i}_valid,\n"

  for i in range(size):
      valid_instance_assignments += (f"    ins_ready({i}) => ins{i}_ready,\n")

  valid_instance_assignments = valid_instance_assignments.lstrip()


  fork_instance_assignments = ""
  for i in range(size):
      fork_instance_assignments += f"    outs_valid({i}) => outs{i}_valid,\n"

  for i in range(size):
      fork_instance_assignments += (f"    outs_ready({i}) => outs{i}_ready,\n")

  fork_instance_assignments = fork_instance_assignments.lstrip()

  assignments = ""
  for i, bitwidth in enumerate(bitwidths):
      potential_assignment = f"  outs{i} <= ins{i};\n"
      assignments += data(potential_assignment, bitwidth)
  assignments = assignments.lstrip()


  architecture = f"""
  -- Architecture of synchronizer
  architecture arch of {name} is
    signal join_valid : std_logic;
    signal fork_ready : std_logic;
  begin
    join : entity work.{join_name}
    port map(
      -- input channels
      {valid_instance_assignments}
      -- output channel to eager fork
      outs_valid => join_valid,
      outs_ready => fork_ready
    );

    fork : entity work.{fork_name}
    port map(
      -- output channels
      {fork_instance_assignments}
      -- input channel from fork
      ins_valid => join_valid,
      ins_ready => fork_ready
    );

    {assignments}

  end architecture;
  """


  return dependencies + entity + architecture


def input_port(i, bitwidth):
    potential_port_declaration = f"ins{i}        : in std_logic_vector({bitwidth} - 1 downto 0);"
    return f"""    -- input port {i}
    {data(potential_port_declaration, bitwidth)}
    ins{i}_valid  : in std_logic;
    ins{i}_ready  : out std_logic;

"""

def output_port(i, bitwidth):
    potential_port_declaration = f"outs{i}        : out std_logic_vector({bitwidth} - 1 downto 0);"
    return f"""    -- output port {i}
    {data(potential_port_declaration, bitwidth)}
    outs{i}_valid  : out std_logic;
    outs{i}_ready  : in std_logic;

"""