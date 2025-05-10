from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.types import Channel, ExtraSignals
from .utils.concat import ConcatLayout
from .utils.generation import generate_signal_wise_forwarding, generate_concat, generate_slice, generate_default_mappings, enumerate_channel_names


def _generate_transfer_logic(in_channels: list[Channel], out_channels: list[Channel]) -> tuple[str, str]:
  first_in_channel_name = in_channels[0]["name"]
  first_out_channel_name = out_channels[0]["name"]

  return f"""transfer_in <= {first_in_channel_name}_valid and {first_in_channel_name}_ready;
  transfer_out <= {first_out_channel_name}_valid and {first_out_channel_name}_ready;""", \
      "  signal transfer_in, transfer_out : std_logic;"


def _generate_forwarding(in_channel_names: list[str], signal_name: str, signal_bitwidth: int, forwarding_assignments: list[str], forwarding_decls: list[str]):
  # Signal-wise forwarding of extra signals from in_channels to `forwarded`
  assignments, decls = generate_signal_wise_forwarding(
      in_channel_names, ["forwarded"], signal_name, signal_bitwidth)
  forwarding_assignments.extend(assignments)
  # Declare extra signals of `forwarded` channel
  forwarding_decls.extend(decls["forwarded"])


def _generate_concat(concat_layout: ConcatLayout) -> tuple[str, str]:
  concat_assignments = []
  concat_decls = []
  # Concatenate `forwarded` extra signals to create `signals_pre_buffer`
  assignments, decls = generate_concat(
      "forwarded", 0, "signals_pre_buffer", concat_layout)
  concat_assignments.extend(assignments)
  # Declare `signals_pre_buffer` data signal
  concat_decls.extend(decls["signals_pre_buffer"])

  return "\n  ".join(concat_assignments), "\n  ".join(concat_decls)


def _generate_slice(concat_layout: ConcatLayout) -> tuple[str, str]:
  slice_assignments = []
  slice_decls = []

  # Slice `signals_post_buffer` to create `sliced` data and extra signals
  assignments, decls = generate_slice(
      "signals_post_buffer", "sliced", 0, concat_layout)
  slice_assignments.extend(assignments)
  # Declare both `signals_post_buffer` data signal and `sliced` data and extra signals
  slice_decls.extend(decls["signals_post_buffer"])
  slice_decls.extend(decls["sliced"])

  return "\n  ".join(slice_assignments), "\n  ".join(slice_decls)


def generate_buffered_signal_manager(
    name: str,
    in_channels: list[Channel],
    out_channels: list[Channel],
    extra_signals: ExtraSignals,
    generate_inner: Callable[[str], str],
    latency: int
) -> str:
  """
  Generate a signal manager architecture that buffers extra signals
  between input and output channels using a FIFO.

  The buffering allows the extra signals (e.g., `spec`, `tag`) to be delayed
  in sync with data paths. Signals are packed into a single bus, stored in the FIFO,
  then unpacked for output.

  Args:
    name: Name of the signal manager entity
    in_channels: List of input channels
    out_channel: List of output channels
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

  entity = generate_entity(name, in_channels, out_channels)

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
  transfer_assignments, transfer_decls = _generate_transfer_logic(
      in_channels, out_channels)

  in_channel_names = enumerate_channel_names(in_channels)
  out_channel_names = enumerate_channel_names(out_channels)

  forwarding_assignments = []
  forwarding_decls = []
  for signal_name, signal_bitwidth in extra_signals.items():
    _generate_forwarding(in_channel_names, signal_name, signal_bitwidth,
                         forwarding_assignments, forwarding_decls)

  concat_assignments, concat_decls = _generate_concat(concat_layout)
  slice_assignments, slice_decls = _generate_slice(concat_layout)

  # Assign the extra signals of `sliced` to the output channel
  output_assignments = []
  for out_channel_name in out_channel_names:
    for signal_name, _ in extra_signals.items():
      output_assignments.append(
          f"{out_channel_name}_{signal_name} <= sliced_{signal_name};")

  # Map channels to inner component
  mappings = generate_default_mappings(in_channels + out_channels)

  architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  {"\n  ".join(forwarding_decls)}
  {concat_decls}
  {slice_decls}
  {transfer_decls}
begin
  -- Transfer signal assignments
  {transfer_assignments}

  -- Forward extra signals
  {"\n  ".join(forwarding_assignments)}

  -- Concat/split extra signals for buffer input/output
  {concat_assignments}
  {slice_assignments}

  -- Assign extra signals to output channels
  {"\n  ".join(output_assignments)}

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
      ins => signals_pre_buffer,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => signals_post_buffer,
      outs_valid => open,
      outs_ready => transfer_out
    );
end architecture;
"""

  return inner + buff + entity + architecture
