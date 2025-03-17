def generate_spec_commit_v2(name, params):
  bitwidth = params["bitwidth"]

  template = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_commit_v2
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    [DATA_SIGNALS]

    dataIn_valid : in std_logic;
    dataIn_ready : out std_logic;
    dataIn_spec : in std_logic_vector(0 downto 0);

    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;

    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic
  );
end entity;

-- Architecture of spec_commit_v2
architecture arch of {name} is
  signal kill_mode : std_logic;
begin
  dataIn_ready <= (not dataIn_spec(0) and dataOut_ready) or
                  kill_mode or
                  (dataIn_spec(0) and ctrl_valid);

  -- Assume that ctrl comes only when dataIn is spec
  ctrl_ready <= dataIn_valid;

  [DATA_ASSIGNMENTS]
  dataOut_valid <= (not dataIn_spec(0) and dataIn_valid) or
                   (dataIn_spec(0) and not kill_mode and (not ctrl) and ctrl_valid);

  kill_mode_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        kill_mode <= '0';
      else
        if ctrl and ctrl_valid then
          kill_mode <= '1';
        elsif dataIn_valid and not dataIn_spec(0) then
          kill_mode <= '0';
        else
          kill_mode <= kill_mode;
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

  if bitwidth > 0:
    template = template.replace("  [DATA_SIGNALS]", f"""
    dataIn : in std_logic_vector({bitwidth} - 1 downto 0);
    dataOut : out std_logic_vector({bitwidth} - 1 downto 0);
""")
    template = template.replace("  [DATA_ASSIGNMENTS]", f"""
  dataOut <= dataIn;
""")
  else:
    template = template.replace("  [DATA_SIGNALS]", "")
    template = template.replace("  [DATA_ASSIGNMENTS]", "")

  return template
