from collections.abc import Callable
from .entity import generate_entity
from .forwarding import forward_extra_signals
from .types import Port, ExtraSignals
from .mapping import generate_simple_inner_port_mappings
from .concat import ConcatenationInfo


def _generate_buffered_transfer_logic(in_ports: list[Port], out_ports: list[Port], transfer_in_name: str, transfer_out_name: str) -> str:
  first_in_port_name = in_ports[0]["name"]
  first_out_port_name = out_ports[0]["name"]

  return f"""
  {transfer_in_name} <= {first_in_port_name}_valid and {first_in_port_name}_ready;
  {transfer_out_name} <= {first_out_port_name}_valid and {first_out_port_name}_ready;""".lstrip()


def _generate_buffered_signal_assignments(in_ports: list[Port], out_ports: list[Port], concat_info: ConcatenationInfo, extra_signals: ExtraSignals, buff_in_name: str, buff_out_name: str) -> str:
  """
  e.g., buff_in(0 downto 0) <= lhs_spec or rhs_spec;
  """
  forwarded_extra_signals = forward_extra_signals(
      extra_signal_names=list(extra_signals),
      in_port_names=[port["name"] for port in in_ports])

  # Concat/split extra signals for buffer input/output.
  signal_assignments = []

  # Generate assignments from individual extra signals to single concatenated variable.
  for signal_name, (msb, lsb) in concat_info.mapping:
    # Concat extra signals for buffer input.
    signal_assignments.append(
        f"  {buff_in_name}({msb} downto {lsb}) <= {forwarded_extra_signals[signal_name]};")

    # Assign extra signals to all output ports
    for out_port in out_ports:
      port_name = out_port["name"]

      # Split extra signals from buffer output.
      signal_assignments.append(
          f"  {port_name}_{signal_name} <= {buff_out_name}({msb} downto {lsb});")

  return "\n".join(signal_assignments).lstrip()


def generate_buffered_signal_manager(name: str, in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals, generate_inner: Callable[[str], str], latency: int):
  # Delayed import to avoid circular dependency
  from generators.handshake.ofifo import generate_ofifo

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  # Generate buffer to store (concatenated) extra signals
  buff_name = f"{name}_buff"
  buff = generate_ofifo(buff_name, {
      "num_slots": latency,
      "bitwidth": extra_signals_bitwidth
  })

  # Generate transfer logic
  transfer_in_name = "transfer_in"
  transfer_out_name = "transfer_out"
  transfer_logic = _generate_buffered_transfer_logic(
      in_ports, out_ports, transfer_in_name, transfer_out_name)

  buff_in_name = "buff_in"
  buff_out_name = "buff_out"
  signal_assignments = _generate_buffered_signal_assignments(
      in_ports, out_ports, concat_info, extra_signals, buff_in_name, buff_out_name)

  mappings = generate_simple_inner_port_mappings(in_ports + out_ports)

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
