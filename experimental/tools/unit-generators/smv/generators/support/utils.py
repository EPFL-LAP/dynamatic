import re


def mlir_type_to_smv_type(type):
  pattern = r"^!handshake\.channel<i(\d+)>$"
  match = re.match(pattern, type)

  if match:
    type_width = int(match.group(1))
    if type_width == 1:
      return "boolean"
    else:
      return f"unsigned word [{type_width}]"
  else:
    raise ValueError(f"Type {type} doesn't correspond to any SMV type")


def smv_init_data_type(smv_type):
  pattern = r"^unsigned word \[(\d+)\]$"
  match = re.match(pattern, smv_type)

  if smv_type == "boolean":
    return "FALSE"
  elif match:
    return 0
  else:
    raise ValueError(f"Type {smv_type} isn't supported")
