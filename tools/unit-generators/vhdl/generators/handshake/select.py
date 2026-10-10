from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.generation import generate_concat_and_handshake, generate_slice_and_handshake, generate_signal_wise_forwarding
from generators.support.signal_manager.utils.internal_signal import create_internal_channel_decl


def generate_select(name, parameters):
    bitwidth = parameters["bitwidth"]
    extra_signals = parameters["extra_signals"]
    # Maximum number of tokens that may be owed to be killed on one data input,
    # i.e., how far one data input may run ahead of the other
    antitoken_depth = parameters.get("antitoken_depth", 1)
    if antitoken_depth < 1:
        raise ValueError(
            f"antitoken_depth must be at least 1, got {antitoken_depth}")

    if extra_signals:
        return _generate_select_signal_manager(name, bitwidth, extra_signals,
                                               antitoken_depth)
    else:
        return _generate_select(name, bitwidth, antitoken_depth)


def _generate_select(name, bitwidth, antitoken_depth):
    cnt_bitwidth = antitoken_depth.bit_length()

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
-- trueValue is selected when condition = 1, falseValue when condition = 0.
-- When the selector fires with one value while the token of the other value has
-- not arrived yet, that token is owed to be killed when it arrives. Kills are
-- owed to at most one of trueValue and falseValue at a time: a condition
-- selecting a value waits at the head of the condition channel until all kills
-- owed to that value are done, so the next token of that value is the one it
-- selects.
--   NORMAL     : no kill owed (cnt = 0)
--   KILL_TRUE  : cnt trueValue tokens owed to be killed, falseValue may run ahead
--   KILL_FALSE : cnt falseValue tokens owed to be killed, trueValue may run ahead
architecture arch of {name} is
  type state_t is (NORMAL, KILL_TRUE, KILL_FALSE);
  signal state                       : state_t;
  signal cnt                         : unsigned({cnt_bitwidth} - 1 downto 0);
  signal killTrueOwed, killFalseOwed : std_logic;
  signal full                        : std_logic;
  signal selTrue, selFalse           : std_logic;
  signal canTrue, canFalse           : std_logic;
  signal fireTrue, fireFalse         : std_logic;
  signal discardTrue, discardFalse   : std_logic;
begin

  killTrueOwed  <= '1' when state = KILL_TRUE else '0';
  killFalseOwed <= '1' when state = KILL_FALSE else '0';
  full          <= '1' when cnt = {antitoken_depth} else '0';

  selTrue  <= condition_valid and condition(0);
  selFalse <= condition_valid and not condition(0);

  -- Select a value if its next token is not owed a kill, and if firing does not
  -- add a kill to the other value when that one is full
  canTrue  <= selTrue and trueValue_valid and not killTrueOwed and
              not (killFalseOwed and full and not falseValue_valid);
  canFalse <= selFalse and falseValue_valid and not killFalseOwed and
              not (killTrueOwed and full and not trueValue_valid);

  fireTrue  <= canTrue and result_ready;
  fireFalse <= canFalse and result_ready;

  -- Kill an arriving token if a kill is owed to its value, or if the other value
  -- is selected in the same cycle (normal join)
  discardTrue  <= trueValue_valid and (killTrueOwed or fireFalse);
  discardFalse <= falseValue_valid and (killFalseOwed or fireTrue);

  result_valid     <= canTrue or canFalse;
  trueValue_ready  <= (not trueValue_valid) or fireTrue or discardTrue;
  falseValue_ready <= (not falseValue_valid) or fireFalse or discardFalse;
  condition_ready  <= (not condition_valid) or fireTrue or fireFalse;

  result <= falseValue when (condition(0) = '0') else
            trueValue;

  -- State and counter update:
  --   - NORMAL -> KILL_TRUE with cnt = 1 when falseValue is selected and no
  --     trueValue token is present to kill in the same cycle.
  --   - KILL_TRUE: cnt + 1 when falseValue is selected and no trueValue token
  --     is present, cnt - 1 when a trueValue token is killed and falseValue is
  --     not selected, unchanged otherwise. Back to NORMAL when cnt reaches 0.
  --   - NORMAL -> KILL_FALSE and KILL_FALSE are symmetric.
  process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state <= NORMAL;
        cnt   <= (others => '0');
      else
        case state is
          when NORMAL =>
            if fireFalse = '1' and trueValue_valid = '0' then
              state <= KILL_TRUE;
              cnt   <= to_unsigned(1, cnt'length);
            elsif fireTrue = '1' and falseValue_valid = '0' then
              state <= KILL_FALSE;
              cnt   <= to_unsigned(1, cnt'length);
            end if;
          when KILL_TRUE =>
            if fireFalse = '1' and trueValue_valid = '0' then
              cnt <= cnt + 1;
            elsif fireFalse = '0' and trueValue_valid = '1' then
              cnt <= cnt - 1;
              -- cnt still holds the old value: the last owed token is killed
              -- and cnt becomes 0 together with the move to NORMAL
              if cnt = 1 then
                state <= NORMAL;
              end if;
            end if;
          when KILL_FALSE =>
            if fireTrue = '1' and falseValue_valid = '0' then
              cnt <= cnt + 1;
            elsif fireTrue = '0' and falseValue_valid = '1' then
              cnt <= cnt - 1;
              -- cnt still holds the old value: the last owed token is killed
              -- and cnt becomes 0 together with the move to NORMAL
              if cnt = 1 then
                state <= NORMAL;
              end if;
            end if;
        end case;
      end if;
    end if;
  end process;

end architecture;
"""

    return entity + architecture


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


def _generate_select_signal_manager(name, bitwidth, extra_signals, antitoken_depth):
    # Layout info for how extra signals are packed into one std_logic_vector
    concat_layout = ConcatLayout(extra_signals)
    extra_signals_total_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_select(inner_name, bitwidth +
                             extra_signals_total_bitwidth, antitoken_depth)

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
