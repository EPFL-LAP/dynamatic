# See docs/Specs/SignalManager.md

from collections.abc import Callable
from .types import Port, ExtraSignals
from .normal import generate_normal_signal_manager
from .buffered import generate_buffered_signal_manager
from .concat import generate_concat_signal_manager
from .bbmerge import generate_bbmerge_signal_manager


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
    ignore_ports = params.get("ignore_ports", [])
    signal_manager = generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner)
  elif type == "bbmerge":
    size = params["size"]
    data_in_name = params["data_in_name"]
    index_name = params["index_name"]
    out_extra_signals = params["out_extra_signals"]
    index_extra_signals = params["index_extra_signals"]
    index_dir = params["index_dir"]
    spec_inputs = params["spec_inputs"]
    signal_manager = generate_bbmerge_signal_manager(
        name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, index_extra_signals, index_dir, spec_inputs, generate_inner)
  else:
    raise ValueError(f"Unsupported signal manager type: {type}")

  return signal_manager + debugging_info
