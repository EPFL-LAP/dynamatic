import re


def mlir_type_to_smv_type(type):
  pattern = r"^!handshake\.channel<([u]?i)(\d+)>$"
  match = re.match(pattern, type)

  if match:
    signed = not match.group(1).startswith("u")
    type_width = int(match.group(2))
    if type_width == 1:
      return "boolean"
    elif signed:
      return f"signed word [{type_width}]"
    else:
      return f"unsigned word [{type_width}]"
  else:
    raise ValueError(f"Type {type} doesn't correspond to any SMV type")


def smv_format_constant(const_value, smv_type):
  pattern = r"^[un]?signed word \[(\d+)\]$"
  match = re.match(pattern, smv_type)

  if smv_type == "boolean":
    return "TRUE" if bool(const_value) else "FALSE"
  elif match:
    return int(const_value)
  else:
    raise ValueError(f"Type {smv_type} isn't supported")
