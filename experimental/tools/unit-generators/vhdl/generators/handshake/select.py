from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_signal_wise_forwarding
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl


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


def _generate_concat(bitwidth: int, concat_layout: ConcatLayout):
    concat_decls = []
    concat_assignments = []

    # Declare trueValue_inner and falseValue_inner channels
    # Example:
    # signal trueValue_inner : std_logic_vector(32 downto 0);
    # signal trueValue_inner_valid : std_logic;
    # signal trueValue_inner_ready : std_logic;
    concat_decls.extend(create_internal_channel_decl({
        "name": "trueValue_inner",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))
    # Example:
    # signal falseValue_inner : std_logic_vector(32 downto 0);
    # signal falseValue_inner_valid : std_logic;
    # signal falseValue_inner_ready : std_logic;
    concat_decls.extend(create_internal_channel_decl({
        "name": "falseValue_inner",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))

    # Concatenate trueValue data and extra signals to create trueValue_inner
    # Example:
    # trueValue_inner(32 - 1 downto 0) <= trueValue;
    # trueValue_inner(32 downto 32) <= trueValue_spec;
    # trueValue_inner_valid <= trueValue_valid;
    # trueValue_ready <= trueValue_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "trueValue", bitwidth, "trueValue_inner", concat_layout))

    # Concatenate falseValue data and extra signals to create falseValue_inner
    # Example:
    # falseValue_inner(32 - 1 downto 0) <= falseValue;
    # falseValue_inner(32 downto 32) <= falseValue_spec;
    # falseValue_inner_valid <= falseValue_valid;
    # falseValue_ready <= falseValue_inner_ready;
    concat_assignments.extend(generate_concat_and_handshake(
        "falseValue", bitwidth, "falseValue_inner", concat_layout))

    return concat_assignments, concat_decls


def _generate_slice(bitwidth: int, concat_layout: ConcatLayout):
    slice_decls = []
    slice_assignments = []

    # Declare both result_inner_concat and result_inner channels
    # Example:
    # signal result_inner_concat : std_logic_vector(32 downto 0);
    # signal result_inner_concat_valid : std_logic;
    # signal result_inner_concat_ready : std_logic;
    slice_decls.extend(create_internal_channel_decl({
        "name": "result_inner_concat",
        "bitwidth": bitwidth + concat_layout.total_bitwidth
    }))
    # Example:
    # signal result_inner : std_logic_vector(31 downto 0);
    # signal result_inner_valid : std_logic;
    # signal result_inner_ready : std_logic;
    # signal result_inner_spec : std_logic_vector(0 downto 0);
    slice_decls.extend(create_internal_channel_decl({
        "name": "result_inner",
        "bitwidth": bitwidth,
        "extra_signals": concat_layout.extra_signals
    }))

    # Slice result_inner_concat to create result_inner data and extra signals
    # Example:
    # result_inner <= result_inner_concat(32 - 1 downto 0);
    # result_inner_spec <= result_inner_concat(32 downto 32);
    # result_inner_valid <= result_inner_concat_valid;
    # result_inner_concat_ready <= result_inner_ready;
    slice_assignments.extend(generate_slice_and_handshake(
        "result_inner_concat", "result_inner", bitwidth, concat_layout))

    return slice_assignments, slice_decls


def _generate_select_signal_manager(name, bitwidth, extra_signals):
    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_total_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_select(inner_name, bitwidth +
                             extra_signals_total_bitwidth)

    entity = generate_entity(name, [{
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
    }], [{
        "name": "result",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }])

    concat_assignments, concat_decls = _generate_concat(
        bitwidth, concat_layout)
    slice_assignments, slice_decls = _generate_slice(
        bitwidth, concat_layout)

    forwarding_assignments = []
    # Signal-wise forwarding of extra signals from condition and result_inner to result
    # Example: result_spec <= condition_spec or result_inner_spec;
    for signal_name in extra_signals:
        forwarding_assignments.extend(generate_signal_wise_forwarding(
            ["condition", "result_inner"], ["result"], signal_name))

    architecture = f"""
-- Architecture of selector signal manager
architecture arch of {name} is
  {"\n  ".join(concat_decls)}
  {"\n  ".join(slice_decls)}
begin
  -- Concatenate extra signals
  {"\n  ".join(concat_assignments)}
  {"\n  ".join(slice_assignments)}

  -- Forwarding logic
  {"\n  ".join(forwarding_assignments)}

  result <= result_inner;
  result_valid <= result_inner_valid;
  result_inner_ready <= result_ready;

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      condition => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueValue => trueValue_inner,
      trueValue_valid => trueValue_inner_valid,
      trueValue_ready => trueValue_inner_ready,
      falseValue => falseValue_inner,
      falseValue_valid => falseValue_inner_valid,
      falseValue_ready => falseValue_inner_ready,
      result => result_inner_concat,
      result_ready => result_inner_concat_ready,
      result_valid => result_inner_concat_valid
    );
end architecture;
"""

    return inner + entity + architecture
