from generators.support.signal_manager import generate_signal_manager


def generate_select(name, parameters):
    bitwidth = parameters["bitwidth"]
    extra_signals = parameters["extra_signals"]

    if extra_signals:
        return _generate_select_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_select(name, bitwidth)


def _generate_antitokens(name):
    return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of antitokens
entity {name} is
  port (
    clk, rst                   : in  std_logic;
    pvalid1, pvalid0           : in  std_logic;
    kill1, kill0               : out std_logic;
    generate_at1, generate_at0 : in  std_logic;
    stop_valid                 : out std_logic
  );
end entity;

-- Architecture of antitokens
architecture arch of {name} is
  signal reg_in0, reg_in1, reg_out0, reg_out1 : std_logic;
begin

  reg0 : process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        reg_out0 <= '0';
      else
        reg_out0 <= reg_in0;
      end if;
    end if;
  end process reg0;

  reg1 : process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        reg_out1 <= '0';
      else
        reg_out1 <= reg_in1;
      end if;
    end if;
  end process reg1;

  reg_in0 <= not pvalid0 and (generate_at0 or reg_out0);
  reg_in1 <= not pvalid1 and (generate_at1 or reg_out1);

  stop_valid <= reg_out0 or reg_out1;

  kill0 <= generate_at0 or reg_out0;
  kill1 <= generate_at1 or reg_out1;
end architecture;
"""


def _generate_select(name, bitwidth):
    antitokens_name = f"{name}_antitokens"
    antitokens = _generate_antitokens(antitokens_name)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of selector
entity {name} is
  port (
    -- inputs
    clk, rst         : in std_logic;
    condition        : in std_logic_vector(0 downto 0);
    condition_valid  : in std_logic;
    trueValue        : in std_logic_vector({bitwidth} - 1 downto 0);
    trueValue_valid  : in std_logic;
    falseValue       : in std_logic_vector({bitwidth} - 1 downto 0);
    falseValue_valid : in std_logic;
    result_ready     : in std_logic;
    -- outputs
    result           : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid     : out std_logic;
    condition_ready  : out std_logic;
    trueValue_ready  : out std_logic;
    falseValue_ready : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of selector
architecture arch of {name} is
  signal ee, validInternal : std_logic;
  signal kill0, kill1      : std_logic;
  signal antitokenStop     : std_logic;
  signal g0, g1            : std_logic;
begin

  ee            <= condition_valid and ((not condition(0) and falseValue_valid) or (condition(0) and trueValue_valid)); --condition(0) and one input
  validInternal <= ee and not antitokenStop; -- propagate ee if not stopped by antitoken

  g0 <= not trueValue_valid and validInternal and result_ready;
  g1 <= not falseValue_valid and validInternal and result_ready;

  result_valid     <= validInternal;
  trueValue_ready  <= (not trueValue_valid) or (validInternal and result_ready) or kill0; -- normal join or antitoken
  falseValue_ready <= (not falseValue_valid) or (validInternal and result_ready) or kill1; --normal join or antitoken
  condition_ready  <= (not condition_valid) or (validInternal and result_ready); --like normal join

  result <= falseValue when (condition(0) = '0') else
            trueValue;

  Antitokens : entity work.{antitokens_name}
    port map(
      clk, rst,
      falseValue_valid, trueValue_valid,
      kill1, kill0,
      g1, g0,
      antitokenStop
    );

end architecture;
"""

    return antitokens + entity + architecture


def _generate_select_signal_manager(name, bitwidth, extra_signals):
    # TODO: Normal signal manager doesn't work for select op.
    # I'll fix it after the refactoring of signal manager functions.
    return generate_signal_manager(name, {
        "type": "normal",
        "in_ports": [{
            "name": "condition",
            "bitwidth": 1,
            "extra_signals": extra_signals
        }, {
            "name": "trueValue",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "falseValue",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        "out_ports": [{
            "name": "result",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        "extra_signals": extra_signals
    }, lambda name: _generate_select(name, bitwidth))
