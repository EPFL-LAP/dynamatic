from .types import Port, ExtraSignals
from .concat import get_default_concat_name


def generate_inner_port_mapping(port: Port, inner_port_data_name: str | None = None, mapping_extra_signals: list[str] = []) -> list[str]:
  """
  Generate VHDL port mappings for a single port.

  If `inner_port_data_name` is None, defaults to the port's own name.
  Only includes extra signals listed in `mapped_extra_signals`.

  Returns a list of string in the form "(inner_signal) => (outer_signal)".
  E.g., ["lhs => lhs", "lhs_valid => lhs_valid", "lhs_ready => lhs_ready", "lhs_<extra> => lhs_<extra>"].
  """
  mapping: list[str] = []
  port_name = port["name"]
  port_extra_signals = port.get("extra_signals", {})

  if inner_port_data_name is None:
    inner_port_data_name = port_name

  if port["bitwidth"] > 0:
    # Mapping for data signal if present
    mapping.append(f"{port_name} => {inner_port_data_name}")

  # Mapping for handshake signals
  mapping.append(f"{port_name}_valid => {port_name}_valid")
  mapping.append(f"{port_name}_ready => {port_name}_ready")

  # Mapping for extra signals
  # Only include extra signals that are in the mapping list
  for signal_name in mapping_extra_signals:
    if signal_name in port_extra_signals:
      mapping.append(
          f"{port_name}_{signal_name} => {port_name}_{signal_name}")

  return mapping


def generate_simple_mappings(ports: list[Port], mapping_extra_signal_names: list[str] = []) -> list[str]:
  """
  Generate simple one-to-one port mappings.
  For each port:
      lhs => lhs,
      lhs_valid => lhs_valid,
      lhs_ready => lhs_ready,
      lhs_<extra> => lhs_<extra>
  """
  mappings = []
  for port in ports:
    mappings += generate_inner_port_mapping(
        port,
        port["name"],
        mapping_extra_signal_names)

  return mappings


def generate_concat_mappings(ports: list[Port], extra_signals_bitwidth: int, mapping_extra_signal_names: list[str] = []) -> list[str]:
  """
  Generate port mappings for concatenated ports.

  The bitwidth is increased by `extra_signals_bitwidth` beyond the original data width.
  Mapped port names are automatically determined using `get_default_concat_name(...)`.
  E.g., `lhs => lhs_inner` (where `lhs_inner` is returned by `get_default_concat_name`).
  """
  mappings = []
  for port in ports:
    port_name = port["name"]

    mapping_port = port.copy()
    mapping_port["bitwidth"] += extra_signals_bitwidth

    mapping_port_name = get_default_concat_name(port_name)

    mappings += generate_inner_port_mapping(
        mapping_port,
        mapping_port_name,
        mapping_extra_signal_names)

  return mappings


def get_unhandled_extra_signals(ports: list[Port], handled_extra_signals: ExtraSignals) -> list[str]:
  """
  Identify extra signals in ports that are not listed in `handled_extra_signals`.

  Returns a list of unhandled extra signal names.
  """
  unhandled_extra_signals: list[str] = []
  for port in ports:
    port_extra_signals = port.get("extra_signals", {})

    for signal_name in port_extra_signals:
      if signal_name not in handled_extra_signals:
        unhandled_extra_signals.append(signal_name)

  return unhandled_extra_signals
