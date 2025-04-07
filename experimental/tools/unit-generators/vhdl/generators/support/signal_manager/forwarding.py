from .types import ExtraSignals, Port


def get_default_extra_signal_value(extra_signal_name: str):
  return "\"0\""


def _get_forwarded_expression(signal_name: str, in_extra_signal_names: list[str]) -> str:
  if signal_name == "spec":
    return " or ".join(in_extra_signal_names)

  raise ValueError(
      f"Unsupported forwarding method for extra signal: {signal_name}")


def forward_extra_signal(extra_signal_name: str, in_port_names: list[str]) -> str:
  if not in_port_names:
    # Use default values for extra signals
    return get_default_extra_signal_value(extra_signal_name)
  else:
    in_extra_signals = []

    # Collect extra signals from all input ports
    for port_name in in_port_names:
      in_extra_signals.append(f"{port_name}_{extra_signal_name}")

    # Forward all input extra signals with the specified method
    return _get_forwarded_expression(extra_signal_name, in_extra_signals)


ForwardingMap = dict[str, str]


def forward_extra_signals(extra_signal_names: list[str], in_port_names: list[str]) -> ForwardingMap:
  """
  Calculate how each extra signal is forwarded to the output ports.
  Result is a dict of extra signal names to VHDL expressions.
  e.g., {"spec": "lhs_spec or rhs_spec", "tag0": "lhs_tag0 (op) rhs_tag0"}
  If no inputs are provided, we use the default values.
  e.g., {"spec": "\"0\"", "tag0": "\"0\""}
  """

  forwarded_extra_signals: dict[str, str] = {}
  # Calculate forwarded extra signals
  for signal_name in extra_signal_names:
    forwarded_extra_signals[signal_name] = forward_extra_signal(
        signal_name, in_port_names)

  return forwarded_extra_signals


def generate_forwarding_assignments_from_forwarding_map(out_port_names: list[str], forwarding_map: ForwardingMap) -> list[str]:
  """
  Generate VHDL assignments for extra signals.
  e.g., "lhs_spec or rhs_spec" => "result_spec <= lhs_spec or rhs_spec;"
  """
  assignments = []
  for out_port_name in out_port_names:
    for signal_name, expression in forwarding_map.items():
      assignments.append(f"  {out_port_name}_{signal_name} <= {expression};")

  return assignments


def generate_forwarding_assignments(in_port_names: list[str], out_port_names: list[str], extra_signal_names: list[str]) -> list[str]:
  forwarding_map = forward_extra_signals(
      extra_signal_names, in_port_names)
  return generate_forwarding_assignments_from_forwarding_map(
      out_port_names, forwarding_map)
