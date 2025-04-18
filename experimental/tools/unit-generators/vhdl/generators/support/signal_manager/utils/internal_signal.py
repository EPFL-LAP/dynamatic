from typing import cast
from .types import Port, ArrayPort


def generate_internal_signal(name: str) -> str:
  return f"signal {name} : std_logic;"


def generate_internal_signal_vector(name: str, bitwidth: int) -> str:
  return f"signal {name} : std_logic_vector({bitwidth - 1} downto 0);"


def generate_internal_signal_array(name: str, bitwidth: int, size: int) -> str:
  return f"signal {name} : data_array({size - 1} downto 0)({bitwidth - 1} downto 0);"


def generate_internal_signals_from_port(port: Port) -> list[str]:
  """
  Generate all internal signal declarations (data, valid/ready, extra signals)
  for a given port. Supports both scalar and array ports.
  """
  name = port["name"]
  bitwidth = port["bitwidth"]
  extra_signals = port.get("extra_signals", {})
  port_array = port.get("array", False)

  signals = []
  if not port_array:
    # Scalar port

    if bitwidth > 0:
      signals.append(generate_internal_signal_vector(name, bitwidth))

    signals.append(generate_internal_signal(f"{name}_valid"))
    signals.append(generate_internal_signal(f"{name}_ready"))

    # Extra signals
    for signal_name, signal_bitwidth in extra_signals.items():
      signals.append(
          generate_internal_signal_vector(f"{name}_{signal_name}", signal_bitwidth))

  else:
    # Array port
    port = cast(ArrayPort, port)
    size = port["size"]

    if bitwidth > 0:
      signals.append(generate_internal_signal_array(name, bitwidth, size))

    signals.append(generate_internal_signal_vector(f"{name}_valid", size))
    signals.append(generate_internal_signal_vector(f"{name}_ready", size))

    # Use per-port extra signal list if available
    use_extra_signals_list = "extra_signals_list" in port

    # Generate extra signal declarations for each item in the 2d input port
    for i in range(size):
      current_extra_signals = (
          port["extra_signals_list"][i] if use_extra_signals_list
          else extra_signals
      )

      # Declare extra signals independently for each array element
      for signal_name, signal_bitwidth in current_extra_signals.items():
        signals.append(
            generate_internal_signal_vector(f"{name}_{i}_{signal_name}", signal_bitwidth))

  return signals
