# See docs/Specs/SignalManager.md

from collections.abc import Callable
from .types import Port, ExtraSignals
from .normal import generate_normal_signal_manager
from .buffered import generate_buffered_signal_manager
from .concat import generate_concat_signal_manager


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]) -> str:
  debugging_info = f"-- Signal manager generation info: {name}, {params}\n"

  in_ports: list[Port] = params["in_ports"]
  out_ports: list[Port] = params["out_ports"]
  type = params["type"]

  if type == "normal":
    extra_signals: ExtraSignals = params["extra_signals"]
    signal_manager = generate_normal_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "buffered":
    extra_signals = params["extra_signals"]
    latency = params["latency"]
    signal_manager = generate_buffered_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner, latency)
  elif type == "concat":
    extra_signals = params["extra_signals"]
    signal_manager = generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals,  generate_inner)
  else:
    raise ValueError(f"Unsupported signal manager type: {type}")

  return signal_manager + debugging_info
