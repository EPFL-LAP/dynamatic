def generate_spec_mux_v2(name, params):
  bitwidth = params["bitwidth"]

  template = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_mux_v2
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;

    [DATA_SIGNALS]

    muxCtrl : in std_logic_vector(1 downto 0);
    muxCtrl_valid : in std_logic;
    muxCtrl_ready : out std_logic;

    nonspecIn_valid : in std_logic;
    nonspecIn_ready : out std_logic;

    specIn_valid : in std_logic;
    specIn_ready : out std_logic;
    specIn_spec : in std_logic_vector(0 downto 0); -- not used

    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic;
    dataOut_spec : out std_logic_vector(0 downto 0)
  );
end entity;

-- Architecture of spec_mux_v2
architecture arch of {name} is
begin
  muxCtrl_ready <= dataOut_ready and ((not muxCtrl(0) and nonspecIn_valid) or (muxCtrl(0) and specIn_valid));
  nonSpecIn_ready <= dataOut_ready and not muxCtrl(0);
  specIn_ready <= dataOut_ready and muxCtrl(0);
  dataOut_valid <= muxCtrl_valid and muxCtrl(1);
  dataOut_spec(0) <= muxCtrl(0);

  [DATA_ASSIGNMENTS]
end architecture;
"""

  if bitwidth > 0:
    template = template.replace("  [DATA_SIGNALS]", f"""
    nonspecIn : in std_logic_vector({bitwidth} - 1 downto 0);
    specIn : in std_logic_vector({bitwidth} - 1 downto 0);
    dataOut : out std_logic_vector({bitwidth} - 1 downto 0);
""")
    template = template.replace("  [DATA_ASSIGNMENTS]", f"""
  dataOut <= specIn when muxCtrl(0) else nonspecIn;
""")
  else:
    template = template.replace("  [DATA_SIGNALS]", "")
    template = template.replace("  [DATA_ASSIGNMENTS]", "")

  return template
