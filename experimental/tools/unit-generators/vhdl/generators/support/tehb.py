from generators.support.utils import VhdlScalarType


def generate_tehb(name, params):
  port_types = params["port_types"]
  ins_type = VhdlScalarType(port_types["ins"])

  if ins_type.is_channel():
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
-- Architecture of tehb_dataless
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
