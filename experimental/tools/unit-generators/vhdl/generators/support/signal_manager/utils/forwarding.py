# Default expression to use when no input ports are present
def get_default_extra_signal_value(extra_signal_name: str):
  """
  Return the default VHDL value for an extra signal
  when there are no input sources to forward from.
  """
  return "\"0\""


def generate_forwarding_expression_for_signal(signal_name: str, in_extra_signal_names: list[str]) -> str:
  """
  Generate a VHDL expression to forward an extra signal
  based on a list of input extra signal names.

  If the list is empty, a default value is returned.
  Currently, only the "spec" signal is supported,
  which is forwarded using a logical OR.
  """

  if not in_extra_signal_names:
    return get_default_extra_signal_value(signal_name)

  if signal_name == "spec":
    return " or ".join(in_extra_signal_names)

  raise ValueError(
      f"Unsupported forwarding method for extra signal: {signal_name}")


def forward_extra_signal(extra_signal_name: str, in_port_names: list[str]) -> str:
  """
  Construct the forwarding expression for a single extra signal
  from all input ports.
  """

  in_extra_signals = [
      f"{port_name}_{extra_signal_name}" for port_name in in_port_names
  ]

  # Forward all input extra signals with the specified method
  return generate_forwarding_expression_for_signal(extra_signal_name, in_extra_signals)


ForwardingMap = dict[str, str]


def forward_extra_signals(extra_signal_names: list[str], in_port_names: list[str]) -> ForwardingMap:
  """
  Generate a map from extra signal name to forwarding VHDL expression.

  For example:
    {"spec": "lhs_spec or rhs_spec"}
    If no inputs: {"spec": "\"0\""}
  """
  forwarded_extra_signals: dict[str, str] = {}

  for signal_name in extra_signal_names:
    forwarded_extra_signals[signal_name] = forward_extra_signal(
        signal_name, in_port_names)

  return forwarded_extra_signals


def generate_forwarding_assignments_from_forwarding_map(out_port_names: list[str], forwarding_map: ForwardingMap) -> list[str]:
  """
  Generate a list of VHDL assignments to drive each extra signal
  on each output port, based on the provided forwarding map.

  Example output:
    result_spec <= lhs_spec or rhs_spec;
  """
  assignments = []

  for out_port_name in out_port_names:
    for signal_name, expression in forwarding_map.items():
      assignments.append(f"{out_port_name}_{signal_name} <= {expression};")

  return assignments


def generate_forwarding_assignments(in_port_names: list[str], out_port_names: list[str], extra_signal_names: list[str]) -> list[str]:
  """
  High-level helper: generate VHDL forwarding assignments
  for a list of extra signals and output ports, based on inputs.
  """

  forwarding_map = forward_extra_signals(
      extra_signal_names, in_port_names)
  return generate_forwarding_assignments_from_forwarding_map(
      out_port_names, forwarding_map)
