from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.types import Port, ExtraSignals
from .utils.concat import ConcatLayout, generate_signal_wise_forwarding, generate_signal_direct_forwarding, generate_concat, generate_slice, subtract_extra_signals, generate_mapping


def _generate_buffered_transfer_logic(
    in_ports: list[Port],
    out_ports: list[Port],
    transfer_in_name: str,
    transfer_out_name: str
) -> str:
  """
  Generate `transfer_in` and `transfer_out` logic based on the valid/ready
  handshake of the first input and output ports.
  """

  first_in_port_name = in_ports[0]["name"]
  first_out_port_name = out_ports[0]["name"]

  return f"""
  {transfer_in_name} <= {first_in_port_name}_valid and {first_in_port_name}_ready;
  {transfer_out_name} <= {first_out_port_name}_valid and {first_out_port_name}_ready;""".lstrip()


def generate_buffered_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str],
    latency: int
) -> str:
  """
  Generate a signal manager architecture that buffers extra signals
  between input and output ports using a FIFO.

  The buffering allows the extra signals (e.g., `spec`, `tag`) to be delayed
  in sync with data paths. Signals are packed into a single bus, stored in the FIFO,
  then unpacked for output.

  Args:
    name: Name of the signal manager entity
    in_ports: List of input ports
    out_ports: List of output ports
    extra_signals: Dictionary of extra signals (e.g., spec, tag) to be handled
    generate_inner: Function to generate the inner component
    latency: FIFO depth

  Returns:
    A string representing the complete VHDL architecture for the signal manager.
  """
  # Delayed import to avoid circular dependency
  from generators.handshake.ofifo import generate_ofifo

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(extra_signals)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  # Generate FIFO to buffer concatenated extra signals
  buff_name = f"{name}_buff"
  buff = generate_ofifo(buff_name, {
      "num_slots": latency,
      "bitwidth": extra_signals_bitwidth
  })

  # Generate transfer handshake logic
  transfer_in_name = "transfer_in"
  transfer_out_name = "transfer_out"
  transfer_logic = _generate_buffered_transfer_logic(
      in_ports, out_ports, transfer_in_name, transfer_out_name)

  # Assign extra signals to FIFO input/output
  # buff_in_name = "buff_in"
  # buff_out_name = "buff_out"
  # signal_assignments = _generate_buffered_signal_assignments(
  #     in_ports, out_ports, concat_layout, extra_signals, buff_in_name, buff_out_name)
  in_channel_names = [port["name"] for port in in_ports]
  forwarded_name = "forwarded"
  forwarded_assignments = []
  forwarded_decls = []
  for signal_name, signal_bitwidth in extra_signals.items():
    assignments, decls = generate_signal_wise_forwarding(
        in_channel_names, [forwarded_name], signal_name, signal_bitwidth)
    forwarded_assignments.extend(assignments)
    forwarded_decls.extend(decls["out"])
  forwarded_assignments = "\n  ".join(forwarded_assignments)
  forwarded_decls = "\n  ".join(forwarded_decls)

  buff_in_name = "buff_in"
  concat_assignments = []
  concat_decls = []
  assignments, decls = generate_concat(
      forwarded_name, 0, buff_in_name, concat_layout)
  concat_assignments.extend(assignments)
  concat_decls.extend(decls["out"])
  concat_assignments = "\n  ".join(concat_assignments)
  concat_decls = "\n  ".join(concat_decls)

  buff_out_name = "buff_out"
  sliced_name = "sliced"
  slice_assignments = []
  slice_decls = []
  assignments, decls = generate_slice(
      buff_out_name, sliced_name, 0, concat_layout)
  slice_assignments.extend(assignments)
  slice_decls.extend(decls["in"])
  slice_decls.extend(decls["out"])
  for signal_name, signal_bitwidth in extra_signals.items():
    for out_port in out_ports:
      port_name = out_port["name"]
      assignments, _ = generate_signal_direct_forwarding(
          sliced_name, port_name, signal_name, signal_bitwidth)
      slice_assignments.extend(assignments)
  slice_assignments = "\n  ".join(slice_assignments)
  slice_decls = "\n  ".join(slice_decls)

  # Map data ports and untouched extra signals directly to inner component
  mapped_ports: list[Port] = []
  for port in in_ports + out_ports:
    mapped_ports.append({
        "name": port["name"],
        "bitwidth": port["bitwidth"],
        "size": port.get("size", 0),
        "extra_signals": subtract_extra_signals(port.get("extra_signals", {}), extra_signals)
    })

  mappings = []
  for port in mapped_ports:
    mappings.extend(generate_mapping(port, port["name"]))
  mappings = ",\n      ".join(mappings)

  architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  {forwarded_decls}
  {concat_decls}
  {slice_decls}
  signal {transfer_in_name}, {transfer_out_name} : std_logic;
begin
  -- Transfer signal assignments
  {transfer_logic}

  -- Concat/split extra signals for buffer input/output
  {forwarded_assignments}
  {concat_assignments}
  {slice_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );

  -- Generate ofifo to store extra signals
  -- num_slots = {latency}, bitwidth = {extra_signals_bitwidth}
  buff : entity work.{buff_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => {buff_in_name},
      ins_valid => {transfer_in_name},
      ins_ready => open,
      outs => {buff_out_name},
      outs_valid => open,
      outs_ready => {transfer_out_name}
    );
end architecture;
"""

  return inner + buff + entity + architecture
