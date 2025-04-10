from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.forwarding import forward_extra_signals
from .utils.types import Port, ExtraSignals
from .utils.mapping import generate_simple_mappings, get_unhandled_extra_signals
from .utils.concat import ConcatLayout


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


def _generate_buffered_signal_assignments(
    in_ports: list[Port],
    out_ports: list[Port],
    concat_layout: ConcatLayout,
    extra_signals: ExtraSignals,
    buff_in_name: str,
    buff_out_name: str
) -> str:
  """
  Generate assignments for buffering extra signals:
  - Forward and concat extra signals from all inputs into a single `buff_in` bus.
  - Split `buff_out` bus to drive extra signals for all outputs.

  Example:
    buff_in(0 downto 0) <= lhs_spec or rhs_spec;
    buff_in(8 downto 1) <= lhs_tag0;
    out_spec <= buff_out(0 downto 0);
    out_tag0 <= buff_out(8 downto 1);
  """
  forwarded_extra_signals = forward_extra_signals(
      extra_signal_names=list(extra_signals),
      in_port_names=[port["name"] for port in in_ports])

  signal_assignments = []

  for signal_name, (msb, lsb) in concat_layout.mapping:
    # Forward signals from multiple input ports into one concatenated vector
    signal_assignments.append(
        f"{buff_in_name}({msb} downto {lsb}) <= {forwarded_extra_signals[signal_name]};")

    # Distribute split buffer outputs to all output ports
    for out_port in out_ports:
      port_name = out_port["name"]
      signal_assignments.append(
          f"{port_name}_{signal_name} <= {buff_out_name}({msb} downto {lsb});")

  return "\n  ".join(signal_assignments)


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
  buff_in_name = "buff_in"
  buff_out_name = "buff_out"
  signal_assignments = _generate_buffered_signal_assignments(
      in_ports, out_ports, concat_layout, extra_signals, buff_in_name, buff_out_name)

  # Map data ports and untouched extra signals directly to inner component
  unhandled_extra_signals = get_unhandled_extra_signals(
      in_ports + out_ports, extra_signals)
  mappings = ",\n      ".join(generate_simple_mappings(
      in_ports + out_ports, unhandled_extra_signals))

  architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  signal {buff_in_name}, {buff_out_name} : std_logic_vector({extra_signals_bitwidth} - 1 downto 0);
  signal {transfer_in_name}, {transfer_out_name} : std_logic;
begin
  -- Transfer signal assignments
  {transfer_logic}

  -- Concat/split extra signals for buffer input/output
  {signal_assignments}

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
