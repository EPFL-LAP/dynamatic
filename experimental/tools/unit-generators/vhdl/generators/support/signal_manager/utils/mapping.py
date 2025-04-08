from .types import Port, ExtraSignals


def generate_inner_port_mapping(port: Port, inner_port_data_name: str | None = None, mapping_extra_signals: list[str] = []) -> list[str]:
  mapping: list[str] = []
  port_name = port["name"]

  if inner_port_data_name is None:
    inner_port_data_name = port_name

  if port["bitwidth"] > 0:
    mapping.append(f"      {port_name} => {inner_port_data_name}")

  mapping.append(f"      {port_name}_valid => {port_name}_valid")
  mapping.append(f"      {port_name}_ready => {port_name}_ready")

  for signal_name in mapping_extra_signals:
    if signal_name in port.get("extra_signals", {}):
      mapping.append(
          f"      {port_name}_{signal_name} => {port_name}_{signal_name}")

  return mapping


def generate_simple_mappings(ports: list[Port], mapping_extra_signal_names: list[str] = []) -> str:
  """
  Generate port forwarding for inner entity
  e.g.,
      lhs => lhs,
      lhs_valid => lhs_valid,
      lhs_ready => lhs_ready
  """
  mappings = []
  for port in ports:
    mappings += generate_inner_port_mapping(
        port,
        port["name"],
        mapping_extra_signal_names)

  return ",\n".join(mappings).lstrip()


def generate_concat_mappings(ports: list[Port], extra_signals_bitwidth: int, mapping_extra_signal_names: list[str] = []) -> str:
  mappings = []
  for port in ports:
    port_name = port["name"]

    mapping_port = port.copy()
    mapping_port["bitwidth"] += extra_signals_bitwidth

    mapping_port_name = f"{port_name}_inner"

    mappings += generate_inner_port_mapping(
        mapping_port,
        mapping_port_name,
        mapping_extra_signal_names)

  return ",\n".join(mappings).lstrip()


def get_unhandled_extra_signals(ports: list[Port], handled_extra_signals: ExtraSignals) -> list[str]:
  unhandled_extra_signals: list[str] = []
  for port in ports:
    port_extra_signals = port.get("extra_signals", {})

    for signal_name in port_extra_signals:
      if signal_name not in handled_extra_signals:
        unhandled_extra_signals.append(signal_name)

  return unhandled_extra_signals
