from generators.support.utils import VhdlScalarType, generate_extra_signal_ports
from generators.support.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.oehb import generate_oehb
from generators.handshake.ofifo import generate_ofifo

def generate_muli(name, params):
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["result"])

  if data_type.has_extra_signals():
    return _generate_muli_signal_manager(name, data_type)
  else:
    return _generate_muli(name, data_type.bitwidth)

def _get_latency():
  return 4

def _generate_mul_4_stage(name, bitwidth):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mul_4_stage
entity {name} is
  port (
    clk : in  std_logic;
    ce  : in  std_logic;
    a   : in  std_logic_vector({bitwidth} - 1 downto 0);
    b   : in  std_logic_vector({bitwidth} - 1 downto 0);
    p   : out std_logic_vector({bitwidth} - 1 downto 0));
end entity;
"""

  architecture = f"""
-- Architecture of mul_4_stage
architecture behav of {name} is

  signal a_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal b_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal q0    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q1    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q2    : std_logic_vector({bitwidth} - 1 downto 0);
  signal mul   : std_logic_vector({bitwidth} - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), {bitwidth}));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;
end architecture;
"""

  return entity + architecture

def _generate_muli(name, bitwidth):
  join_name = f"{name}_join"
  mul_4_stage_name = f"{name}_mul_4_stage"
  buff_name = f"{name}_buff"
  oehb_name = f"{name}_oehb"

  dependencies = \
    generate_join(join_name, 2) + \
    _generate_mul_4_stage(mul_4_stage_name, bitwidth) + \
    generate_delay_buffer(buff_name, _get_latency() - 1) + \
    generate_oehb(oehb_name, {
      "port_types": {
        "ins": f"!handshake.channel<i{bitwidth}>",
        "outs": f"!handshake.channel<i{bitwidth}>"
      }
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of muli
entity {name} is
  port (
    -- inputs
    clk, rst     : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of muli
architecture arch of {name} is
  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector({bitwidth} - 1 downto 0);
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  multiply_unit : entity work.{mul_4_stage_name}(behav)
    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => lhs,
      b   => rhs,
      p   => result
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.{oehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );
end architecture;
"""

  return dependencies + entity + architecture

def _generate_muli_signal_manager(name, data_type):
  inner_name = f"{name}_inner"

  bitwidth = data_type.bitwidth

  dependencies = _generate_muli(inner_name, bitwidth)

  if "spec" in data_type.extra_signals:
    dependencies += generate_ofifo(f"{name}_spec_ofifo", {
      "num_slots": _get_latency(), # todo: correct?
      "port_types": {
        "ins": "!handshake.channel<i1>",
        "outs": "!handshake.channel<i1>"
      }
    })

  # Now that the logic depends on the name, this dict is defined inside this function.
  extra_signal_logic = {
    "spec": (
      # First string is for the signal declaration
      """
    signal spec_tfifo_in : std_logic_vector(0 downto 0);
    signal spec_tfifo_out : std_logic_vector(0 downto 0);
""",
      # Second string is for the actual logic
      f"""
    spec_tfifo_in <= lhs_spec or rhs_spec;
    spec_tfifo : entity work.{name}_spec_ofifo(arch)
      port map(
        clk => clk,
        rst => rst,
        ins => spec_tfifo_in,
        ins_valid => transfer_in,
        ins_ready => open,
        outs => spec_tfifo_out,
        outs_valid => open,
        outs_ready => transfer_out
      );
    result_spec <= spec_tfifo_out;
""")
  }

  for signal_name in data_type.extra_signals:
    if signal_name not in extra_signal_logic:
      raise ValueError(f"Extra signal {signal_name} is not supported")

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of muli signal manager
entity {name} is
  port (
    [EXTRA_SIGNAL_PORTS]
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(32 - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(32 - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(32 - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("lhs", "in"), ("rhs", "in"),
    ("result", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of muli signal manager
architecture arch of {name} is
  signal transfer_in, transfer_out : std_logic;
  [EXTRA_SIGNAL_SIGNAL_DECLS]
begin
  transfer_in <= lhs_valid and lhs_ready;
  transfer_out <= result_valid and result_ready;

  -- list of logic for supported extra signals
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      lhs => lhs,
      lhs_valid => lhs_valid,
      rhs => rhs,
      rhs_valid => rhs_valid,
      result_ready => result_ready,
      result => result,
      result_valid => result_valid,
      lhs_ready => lhs_ready,
      rhs_ready => rhs_ready
    );
end architecture;
"""

  architecture = architecture.replace("  [EXTRA_SIGNAL_SIGNAL_DECLS]",
    "\n".join([
      extra_signal_logic[name][0] for name in data_type.extra_signals
    ]))
  architecture = architecture.replace("  [EXTRA_SIGNAL_LOGIC]",
    "\n".join([
      extra_signal_logic[name][1] for name in data_type.extra_signals
    ]))

  return dependencies + entity + architecture
