from generators.support.arith2 import generate_arith2

def generate_cmpi(name, params):
    bitwidth = params["bitwidth"]
    predicate = params["predicate"]
    extra_signals = params.get("extra_signals", None)

    modifier = _get_sign_from_predicate(predicate)
    comparator = _get_symbol_from_predicate(predicate)

    modType = "cmpi"

    signals = ""

    body = f"""
  result(0) <= '1' when ({modifier}(lhs) {comparator} {modifier}(rhs)) else '0';
"""

    dependencies = ""
    latency = 0

    return generate_arith2(
        name,
        modType,
        bitwidth,
        signals,
        body,
        latency,
        dependencies,
        extra_signals
    )

def _get_symbol_from_predicate(pred):
    match pred:
        case "eq":
            return "="
        case "ne":
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
        case "eq" | "ne":
            return ""
        case "slt" | "sle" | "sgt" | "sge":
            return "signed"
        case "ult" | "ule" | "ugt" | "uge":
            return "unsigned"
        case _:
            raise ValueError(f"Predicate {pred} not known")
