from generators.support.utils import *


def generate_select(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])
    # Maximum number of tokens that may be owed to be killed on one data input,
    # i.e., how far one data input may run ahead of the other
    antitoken_depth = params.get(ATTR_ANTITOKEN_DEPTH, 1)
    if antitoken_depth < 1:
        raise ValueError(
            f"antitoken_depth must be at least 1, got {antitoken_depth}")

    return _generate_select(name, data_type, antitoken_depth)


def _generate_select(name, data_type, antitoken_depth):
    return f"""
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
MODULE {name} (condition, condition_valid, trueValue, trueValue_valid, falseValue, falseValue_valid, result_ready)
  VAR
  state : {{NORMAL, KILL_TRUE, KILL_FALSE}};
  cnt : 0..{antitoken_depth};

  DEFINE
  killTrueOwed := state = KILL_TRUE;
  killFalseOwed := state = KILL_FALSE;
  full := cnt = {antitoken_depth};
  selTrue := condition_valid & condition;
  selFalse := condition_valid & !condition;
  canTrue := selTrue & trueValue_valid & !killTrueOwed & !(killFalseOwed & full & !falseValue_valid);
  canFalse := selFalse & falseValue_valid & !killFalseOwed & !(killTrueOwed & full & !trueValue_valid);
  fireTrue := canTrue & result_ready;
  fireFalse := canFalse & result_ready;
  discardTrue := trueValue_valid & (killTrueOwed | fireFalse);
  discardFalse := falseValue_valid & (killFalseOwed | fireTrue);

  -- State and counter update:
  --   - NORMAL -> KILL_TRUE with cnt = 1 when falseValue is selected and no
  --     trueValue token is present to kill in the same cycle.
  --   - KILL_TRUE: cnt + 1 when falseValue is selected and no trueValue token
  --     is present, cnt - 1 when a trueValue token is killed and falseValue is
  --     not selected, unchanged otherwise. Back to NORMAL when cnt reaches 0.
  --   - NORMAL -> KILL_FALSE and KILL_FALSE are symmetric.
  ASSIGN
  init(state) := NORMAL;
  next(state) := case
    state = NORMAL & fireFalse & !trueValue_valid : KILL_TRUE;
    state = NORMAL & fireTrue & !falseValue_valid : KILL_FALSE;
    -- cnt still holds the old value: the last owed token is killed
    -- and cnt becomes 0 together with the move to NORMAL
    state = KILL_TRUE & !fireFalse & trueValue_valid & cnt = 1 : NORMAL;
    state = KILL_FALSE & !fireTrue & falseValue_valid & cnt = 1 : NORMAL;
    TRUE : state;
  esac;
  init(cnt) := 0;
  -- the cnt guards are redundant in reachable states and only keep cnt in range
  next(cnt) := case
    state = NORMAL & fireFalse & !trueValue_valid : 1;
    state = NORMAL & fireTrue & !falseValue_valid : 1;
    state = KILL_TRUE & fireFalse & !trueValue_valid & cnt < {antitoken_depth} : cnt + 1;
    state = KILL_TRUE & !fireFalse & trueValue_valid & cnt > 0 : cnt - 1;
    state = KILL_FALSE & fireTrue & !falseValue_valid & cnt < {antitoken_depth} : cnt + 1;
    state = KILL_FALSE & !fireTrue & falseValue_valid & cnt > 0 : cnt - 1;
    TRUE : cnt;
  esac;

  -- output
  DEFINE
  trueValue_ready := !trueValue_valid | fireTrue | discardTrue;
  falseValue_ready := !falseValue_valid | fireFalse | discardFalse;
  condition_ready := !condition_valid | fireTrue | fireFalse;
  result_valid := canTrue | canFalse;
  result := condition ? trueValue : falseValue;
"""
