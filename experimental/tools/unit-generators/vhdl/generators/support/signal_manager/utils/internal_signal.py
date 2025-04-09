from typing import cast
from .types import Port, ArrayPort


def generate_internal_signal(name: str) -> str:
  return f"signal {name} : std_logic;"


def generate_internal_signal_vector(name: str, bitwidth: int) -> str:
  return f"signal {name} : std_logic_vector({bitwidth - 1} downto 0);"


def generate_internal_signal_array(name: str, bitwidth: int, size: int) -> str:
  return f"signal {name} : data_array({size - 1} downto 0)({bitwidth - 1} downto 0);"


def generate_internal_signals_from_port(port: Port) -> list[str]:
  name = port["name"]
  bitwidth = port["bitwidth"]
  extra_signals = port.get("extra_signals", {})
  port_array = port.get("array", False)

  signals = []
  if not port_array:
    if bitwidth > 0:
      signals.append(generate_internal_signal_vector(name, bitwidth))

    signals.append(generate_internal_signal(f"{name}_valid"))
    signals.append(generate_internal_signal(f"{name}_ready"))

    # Generate extra signals for this port
    for signal_name, signal_bitwidth in extra_signals.items():
      signals.append(
          generate_internal_signal_vector(f"{name}_{signal_name}", signal_bitwidth))

  else:
    port = cast(ArrayPort, port)
    size = port["size"]

    if bitwidth > 0:
      signals.append(generate_internal_signal_array(name, bitwidth, size))

    signals.append(generate_internal_signal_vector(f"{name}_valid", size))
    signals.append(generate_internal_signal_vector(f"{name}_ready", size))

    # Use extra_signals_list if available to handle per-port extra signals
    use_extra_signals_list = "extra_signals_list" in port

    # Generate extra signal declarations for each item in the 2d input port
    for i in range(size):
      if use_extra_signals_list:
        # Use different extra signals for different ports
        current_extra_signals = port["extra_signals_list"][i]
      else:
        # Use the same extra signals for all items
        current_extra_signals = extra_signals

      # The netlist generator declares extra signals independently for each item,
      # in contrast to ready/valid signals.
      for signal_name, signal_bitwidth in current_extra_signals.items():
        signals.append(
            generate_internal_signal_vector(f"{name}_{i}_{signal_name}", signal_bitwidth))

  return signals
