import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports

def generate_tehb(name, params):
  port_types = ast.literal_eval(params["port_types"])
  ins_type = VhdlScalarType(port_types["ins"])

  if ins_type.has_extra_signals():
    if ins_type.is_channel():
      return _generate_tehb_signal_manager(name, ins_type)
    else:
      return _generate_tehb_signal_manager_dataless(name, ins_type)
  elif ins_type.is_channel():
    return _generate_tehb(name, ins_type.bitwidth)
  else:
    return _generate_tehb_dataless(name)

def _generate_tehb_dataless(name):
  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
architecture arch of {name} is
  signal fullReg, outputValid : std_logic;
begin
  outputValid <= ins_valid or fullReg;

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fullReg <= '0';
      else
        fullReg <= outputValid and not outs_ready;
      end if;
    end if;
  end process;

  ins_ready  <= not fullReg;
  outs_valid <= outputValid;
end architecture;
"""

  return entity + architecture

def _generate_tehb(name, bitwidth):
  tehb_dataless_name = f"{name}_dataless"

  dependencies = _generate_tehb_dataless(tehb_dataless_name)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of tehb
architecture arch of {name} is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector({bitwidth} - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.{tehb_dataless_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => regNotFull,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        dataReg <= (others => '0');
      elsif (regEnable) then
        dataReg <= ins;
      end if;
    end if;
  end process;

  process (regNotFull, dataReg, ins) is
  begin
    if (regNotFull) then
      outs <= ins;
    else
      outs <= dataReg;
    end if;
  end process;

  ins_ready <= regNotFull;

end architecture;
"""

  return dependencies + entity + architecture

def _generate_tehb_signal_manager(name, ins_type):
  inner_name = f"{name}_inner"

  bitwidth = ins_type.bitwidth

  extra_signal_bit_map = {}
  occupied_bits = bitwidth
  for signal_name, signal_bitwidth in ins_type.extra_signals.items():
    extra_signal_bit_map[signal_name] = (
      occupied_bits + signal_bitwidth - 1,
      occupied_bits
    )
    occupied_bits += signal_bitwidth

  full_bitwidth = occupied_bits

  dependencies = _generate_tehb(f"{name}_inner", full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], ins_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of tehb signal manager
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_inner = ["ins"] + [
    "ins_" + name for name in ins_type.extra_signals
  ]
  ins_inner.reverse()
  ins_conversion = f"  ins_inner <= {" & ".join(ins_inner)}\n"

  outs_inner = [
    f"  outs <= outs_inner({bitwidth} - 1 downto 0)"
  ]
  for name in ins_type.extra_signals:
    msb, lsb = extra_signal_bit_map[name]
    outs_inner.append(f"  outs_{name} <= outs_inner({msb} downto {lsb})")
  outs_conversion = "\n".join(outs_inner)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture

def _generate_tehb_signal_manager_dataless(name, ins_type):
  inner_name = f"{name}_inner"

  bitwidth = 0

  extra_signal_bit_map = {}
  occupied_bits = bitwidth
  for signal_name, signal_bitwidth in ins_type.extra_signals.items():
    extra_signal_bit_map[signal_name] = (
      occupied_bits + signal_bitwidth - 1,
      occupied_bits
    )
    occupied_bits += signal_bitwidth

  full_bitwidth = occupied_bits

  dependencies = _generate_tehb(f"{name}_inner", full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], ins_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of tehb signal manager dataless
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_inner = [
    "ins_" + name for name in ins_type.extra_signals
  ]
  ins_inner.reverse()
  ins_conversion = f"  ins_inner <= {" & ".join(ins_inner)}\n"

  outs_inner = []
  for name in ins_type.extra_signals:
    msb, lsb = extra_signal_bit_map[name]
    outs_inner.append(f"  outs_{name} <= outs_inner({msb} downto {lsb})")
  outs_conversion = "\n".join(outs_inner)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture