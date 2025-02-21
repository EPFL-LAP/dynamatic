from generators.support.utils import VhdlScalarType
from generators.support.join import generate_join

def generate_cmpi(name, params):
  port_types = params["port_types"]
  predicate = params["predicate"]
  data_type = VhdlScalarType(port_types["lhs"])

  return _generate_cmpi(name, predicate, data_type.bitwidth)

def _get_symbol_from_predicate(pred):
  match pred:
    case "eq":
      return "="
    case "neq":
      return "/="
    case "slt" | "ult":
      return "<"
    case "sle" | "ule":
      return "<="
    case "sgt" | "ugt":
      return ">"
    case "sge" | "uge":
      return ">="
    case _:
      raise ValueError(f"Predicate {pred} not known")

def _get_sign_from_predicate(pred):
  match pred:
    case "eq" | "neq":
      return ""
    case "slt" | "sle" | "sgt" | "sge":
      return "signed"
    case "ult" | "ule" | "ugt" | "uge":
      return "unsigned"
    case _:
      raise ValueError(f"Predicate {pred} not known")

def _generate_cmpi(name, predicate, bitwidth):
  join_name = f"{name}_join"

  dependencies = generate_join(join_name, {"size": 2})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cmpi
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  modifier = _get_sign_from_predicate(predicate)
  comparator = _get_symbol_from_predicate(predicate)

  architecture = f"""
-- Architecture of cmpi
architecture arch of {name} is
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => result_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  result(0) <= '1' when ({modifier}(lhs) {comparator} {modifier}(rhs)) else '0';
end architecture;
"""

  return dependencies + entity + architecture
