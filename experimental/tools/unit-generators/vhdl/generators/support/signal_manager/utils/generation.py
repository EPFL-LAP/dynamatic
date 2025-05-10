from .types import Channel, ExtraSignals
from .concat import ConcatLayout
from .internal_signal import generate_internal_signal, generate_internal_signal_vector, generate_internal_signal_array
from .forwarding import generate_forwarding_expression_for_signal


Decls = dict[str, list[str]]


def generate_handshake_forwarding(in_channel_name: str, out_channel_name: str, array_size=0) -> tuple[list[str], Decls]:
  assignments = []
  in_declarations = []
  out_declarations = []

  assignments.append(f"{out_channel_name}_valid <= {in_channel_name}_valid;")
  assignments.append(f"{in_channel_name}_ready <= {out_channel_name}_ready;")

  if array_size == 0:
    in_declarations.append(
        generate_internal_signal(f"{in_channel_name}_valid"))
    in_declarations.append(
        generate_internal_signal(f"{in_channel_name}_ready"))
    out_declarations.append(
        generate_internal_signal(f"{out_channel_name}_valid"))
    out_declarations.append(
        generate_internal_signal(f"{out_channel_name}_ready"))
  else:
    in_declarations.append(
        generate_internal_signal_vector(f"{in_channel_name}_valid", array_size))
    in_declarations.append(
        generate_internal_signal_vector(f"{in_channel_name}_ready", array_size))
    out_declarations.append(
        generate_internal_signal_vector(f"{out_channel_name}_valid", array_size))
    out_declarations.append(
        generate_internal_signal_vector(f"{out_channel_name}_ready", array_size))

  decls = {
      in_channel_name: in_declarations,
      out_channel_name: out_declarations
  }
  return assignments, decls


def generate_concat(in_channel_name: str, in_data_bitwidth: int, out_channel_name: str, concat_layout: ConcatLayout, array_size=0) -> tuple[list[str], Decls]:
  assignments = []
  in_declarations = []
  out_declarations = []

  if array_size == 0:
    # Include data if present
    if in_data_bitwidth > 0:
      assignments.append(
          f"{out_channel_name}({in_data_bitwidth} - 1 downto 0) <= {in_channel_name};")
      in_declarations.append(
          generate_internal_signal_vector(in_channel_name, in_data_bitwidth))

    # Include all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      assignments.append(
          f"{out_channel_name}({msb + in_data_bitwidth} downto {lsb + in_data_bitwidth}) <= {in_channel_name}_{signal_name};")
      in_declarations.append(
          generate_internal_signal_vector(f"{in_channel_name}_{signal_name}", msb - lsb + 1))

    out_declarations.append(
        generate_internal_signal_vector(out_channel_name, in_data_bitwidth + concat_layout.total_bitwidth))
  else:
    # Signal is an array
    for i in range(array_size):
      # Include data if present
      if in_data_bitwidth > 0:
        assignments.append(
            f"{out_channel_name}({i})({in_data_bitwidth} - 1 downto 0) <= {in_channel_name}({i});")

      # Include all extra signals
      for signal_name, (msb, lsb) in concat_layout.mapping:
        assignments.append(
            f"{out_channel_name}({i})({msb + in_data_bitwidth} downto {lsb + in_data_bitwidth}) <= {in_channel_name}_{i}_{signal_name};")
        in_declarations.append(
            generate_internal_signal_vector(f"{in_channel_name}_{i}_{signal_name}", msb - lsb + 1))

    in_declarations.append(
        generate_internal_signal_array(in_channel_name, in_data_bitwidth, array_size))
    out_declarations.append(
        generate_internal_signal_array(out_channel_name, in_data_bitwidth + concat_layout.total_bitwidth, array_size))

  decls = {
      in_channel_name: in_declarations,
      out_channel_name: out_declarations
  }
  return assignments, decls


def generate_slice(in_channel_name: str, out_channel_name: str, out_data_bitwidth: int, concat_layout: ConcatLayout, array_size=0) -> tuple[list[str], Decls]:
  assignments = []
  in_declarations = []
  out_declarations = []

  if array_size == 0:
    # Include data if present
    if out_data_bitwidth > 0:
      assignments.append(
          f"{out_channel_name} <= {in_channel_name}({out_data_bitwidth} - 1 downto 0);")
      out_declarations.append(
          generate_internal_signal_vector(out_channel_name, out_data_bitwidth))

    # Include all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      assignments.append(
          f"{out_channel_name}_{signal_name} <= {in_channel_name}({msb + out_data_bitwidth} downto {lsb + out_data_bitwidth});")
      out_declarations.append(
          generate_internal_signal_vector(f"{out_channel_name}_{signal_name}", msb - lsb + 1))

    in_declarations.append(
        generate_internal_signal_vector(in_channel_name, out_data_bitwidth + concat_layout.total_bitwidth))
  else:
    # Signal is an array
    for i in range(array_size):
      # Include data if present
      if out_data_bitwidth > 0:
        assignments.append(
            f"{out_channel_name}({i}) <= {in_channel_name}({i})({out_data_bitwidth} - 1 downto 0);")

      # Include all extra signals
      for signal_name, (msb, lsb) in concat_layout.mapping:
        assignments.append(
            f"{out_channel_name}_{i}_{signal_name} <= {in_channel_name}({i})({msb + out_data_bitwidth} downto {lsb + out_data_bitwidth});")
        out_declarations.append(
            generate_internal_signal_vector(f"{out_channel_name}_{i}_{signal_name}", msb - lsb + 1))

    in_declarations.append(
        generate_internal_signal_array(in_channel_name, out_data_bitwidth + concat_layout.total_bitwidth, array_size))
    out_declarations.append(
        generate_internal_signal_array(out_channel_name, out_data_bitwidth, array_size))

  decls = {
      in_channel_name: in_declarations,
      out_channel_name: out_declarations
  }
  return assignments, decls


def generate_signal_assignment(in_channel_name: str, out_channel_name: str, signal_name: str, signal_bitwidth: int) -> tuple[list[str], Decls]:
  assignments = [
      f"{out_channel_name}_{signal_name} <= {in_channel_name}_{signal_name};"]
  decls = {
      in_channel_name: [generate_internal_signal_vector(
          f"{in_channel_name}_{signal_name}", signal_bitwidth)],
      out_channel_name: [generate_internal_signal_vector(
          f"{out_channel_name}_{signal_name}", signal_bitwidth)]
  }
  return assignments, decls


def generate_mapping(channel: Channel, inner_channel_name: str) -> list[str]:
  mapping: list[str] = []
  channel_name = channel["name"]
  channel_extra_signals = channel.get("extra_signals", {})
  channel_bitwidth = channel["bitwidth"]

  if channel_bitwidth > 0:
    # Mapping for data signal if present
    mapping.append(f"{inner_channel_name} => {channel_name}")

  # Mapping for handshake signals
  mapping.append(f"{inner_channel_name}_valid => {channel_name}_valid")
  mapping.append(f"{inner_channel_name}_ready => {channel_name}_ready")

  for signal_name in channel_extra_signals:
    mapping.append(
        f"{channel_name}_{signal_name} => {channel_name}_{signal_name}")

  return mapping


def generate_default_mappings(channels: list[Channel]) -> str:
  mappings = []
  for channel in channels:
    mappings.extend(generate_mapping({
        **channel,
        # Exclude extra signals from the mapping
        "extra_signals": {}
    }, channel["name"]))
  return ",\n      ".join(mappings)


def enumerate_channel_names(channels: list[Channel]) -> list[str]:
  channel_names = []
  for channel in channels:
    size = channel.get("size", 0)
    if size == 0:
      channel_names.append(channel["name"])
    else:
      for i in range(size):
        channel_names.append(f"{channel['name']}_{i}")
  return channel_names


def generate_signal_wise_forwarding(in_channel_names: list[str], out_channel_names: list[str], extra_signal_name: str, extra_signal_bitwidth: int) -> tuple[list[str], Decls]:
  assignments = []
  decls = {}

  in_extra_signal_names = []
  for in_channel_name in in_channel_names:
    signal_name = f"{in_channel_name}_{extra_signal_name}"
    in_extra_signal_names.append(signal_name)
    decls[in_channel_name] = [
        generate_internal_signal_vector(signal_name, extra_signal_bitwidth)]

  expression = generate_forwarding_expression_for_signal(
      extra_signal_name, in_extra_signal_names)

  for out_channel_name in out_channel_names:
    signal_name = f"{out_channel_name}_{extra_signal_name}"
    assignments.append(f"{signal_name} <= {expression};")
    decls[out_channel_name] = [
        generate_internal_signal_vector(signal_name, extra_signal_bitwidth)]

  return assignments, decls
