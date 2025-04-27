from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.concat import ConcatLayout, generate_concat, generate_slice, generate_mapping, generate_handshake_forwarding
from .utils.types import Port, ExtraSignals


def generate_spec_units_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
    extra_signals_without_spec: ExtraSignals,
    ctrl_names: list[str],
    generate_inner: Callable[[str], str]
):
  """
  Generate a VHDL signal manager for speculative units that handles extra signals
  except for the `spec` signal.

  This function manages the concatenation of extra signals (excluding `spec`) and
  maps them to the inner component while handling tagless control signals
  separately. The control signals are mapped directly to the inner component
  without concatenation.

  Args:
    name: Name for the signal manager entity.
    in_ports: List of input ports for the signal manager.
    out_ports: List of output ports for the signal manager.
    extra_signals_without_spec: List of extra signals (except `spec`) to be handled.
    ctrl_names: List of control signal names that should be separated from data signals.
    generate_inner: Function to generate the inner component.

  Returns:
    A string representing the complete VHDL architecture for the signal manager.
  """

  entity = generate_entity(name, in_ports, out_ports)

  # Separate input ports into control ports and non-control ports
  in_ports_without_ctrl = [
      port for port in in_ports if not port["name"] in ctrl_names]
  ctrl_ports = [
      port for port in in_ports if port["name"] in ctrl_names]

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals_without_spec)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  mapped_ports = {}

  # Declare concatenated signals for all input/output ports excluding control signals
  concat_signal_decls = []
  concat_logic = []
  for port in in_ports_without_ctrl:
    channel_name = port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = port["bitwidth"]
    channel_size = port.get("size", 0)

    assignments, decls = generate_concat(
        channel_name, channel_bitwidth, concat_name, concat_layout, channel_size)
    concat_signal_decls.extend(decls["out"])
    concat_logic.extend(assignments)

    assignments, decls = generate_handshake_forwarding(
        channel_name, concat_name, channel_size)
    concat_signal_decls.extend(decls["out"])
    concat_logic.extend(assignments)

    mapped_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {"spec": 1}
    }

  for port in out_ports:
    channel_name = port["name"]
    concat_name = f"{channel_name}_concat"
    channel_bitwidth = port["bitwidth"]
    channel_size = port.get("size", 0)

    assignments, decls = generate_slice(
        concat_name, channel_name, channel_bitwidth, concat_layout, channel_size)
    concat_signal_decls.extend(decls["in"])
    concat_logic.extend(assignments)

    assignments, decls = generate_handshake_forwarding(
        concat_name, channel_name, channel_size)
    concat_signal_decls.extend(decls["in"])
    concat_logic.extend(assignments)

    mapped_ports[channel_name] = {
        "name": concat_name,
        "bitwidth": channel_bitwidth + concat_layout.total_bitwidth,
        "size": channel_size,
        "extra_signals": {"spec": 1}
    }

  concat_signal_decls = "\n  ".join(concat_signal_decls)
  concat_logic = "\n  ".join(concat_logic)

  # Map control ports, concatenated non-control ports and `spec` signal to inner component
  for ctrl_port in ctrl_ports:
    mapped_ports[ctrl_port["name"]] = ctrl_port

  mappings = []
  for original_name, mapped_channel in mapped_ports.items():
    mappings.extend(generate_mapping(mapped_channel, original_name))
  mappings = ",\n      ".join(mappings)

  architecture = f"""
-- Architecture of signal manager (spec_units)
architecture arch of {name} is
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
