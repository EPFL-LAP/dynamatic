from .types import Channel, ExtraSignals
from .concat import ConcatLayout
from .internal_signal import generate_internal_signal, generate_internal_signal_vector, generate_internal_signal_array
from .forwarding import generate_forwarding_expression_for_signal


def generate_concat(in_channel_name: str, in_data_bitwidth: int, out_channel_name: str, concat_layout: ConcatLayout, array_size=0) -> list[str]:
  """
  Generate VHDL assignments for concatenating input channel data and extra
  signals into output channel data.
  Args:
    in_channel_name (str): Name of the input channel.
    in_data_bitwidth (int): Bitwidth of the input channel data.
    out_channel_name (str): Name of the output channel.
    concat_layout (ConcatLayout): Layout for concatenating extra signals.
    array_size (int): Size of the array if the channel is an array. Defaults to 0.
  Returns:
    assignments (list[str]): List of VHDL assignments for the concatenation.
  """

  assignments = []

  if array_size == 0:
    # Include data if present
    if in_data_bitwidth > 0:
      assignments.append(
          f"{out_channel_name}({in_data_bitwidth} - 1 downto 0) <= {in_channel_name};")

    # Include all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      assignments.append(
          f"{out_channel_name}({msb + in_data_bitwidth} downto {lsb + in_data_bitwidth}) <= {in_channel_name}_{signal_name};")
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

  return assignments


def generate_slice(in_channel_name: str, out_channel_name: str, out_data_bitwidth: int, concat_layout: ConcatLayout, array_size=0) -> list[str]:
  """
  Generate VHDL assignments for slicing input channel data into output channel
  data and extra signals.
  Args:
    in_channel_name (str): Name of the input channel.
    out_channel_name (str): Name of the output channel.
    out_data_bitwidth (int): Bitwidth of the output channel data.
    concat_layout (ConcatLayout): Layout for slicing extra signals.
    array_size (int): Size of the array if the channel is an array. Defaults to 0.
  Returns:
    assignments (list[str]): List of VHDL assignments for the slicing.
  """

  assignments = []

  if array_size == 0:
    # Include data if present
    if out_data_bitwidth > 0:
      assignments.append(
          f"{out_channel_name} <= {in_channel_name}({out_data_bitwidth} - 1 downto 0);")

    # Include all extra signals
    for signal_name, (msb, lsb) in concat_layout.mapping:
      assignments.append(
          f"{out_channel_name}_{signal_name} <= {in_channel_name}({msb + out_data_bitwidth} downto {lsb + out_data_bitwidth});")
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

  return assignments


def generate_concat_and_handshake(in_channel_name: str, in_data_bitwidth: int, out_channel_name: str, concat_layout: ConcatLayout, array_size=0) -> list[str]:
  """
  Generate VHDL assignments for concatenating input channel data and extra
  signals into output channel data, and forwarding the handshake signals.
  Args:
    in_channel_name (str): Name of the input channel.
    in_data_bitwidth (int): Bitwidth of the input channel data.
    out_channel_name (str): Name of the output channel.
    concat_layout (ConcatLayout): Layout for concatenating extra signals.
    array_size (int): Size of the array if the channel is an array. Defaults to 0.
  Returns:
    assignments (list[str]): List of VHDL assignments for the concatenation and handshake forwarding.
  """

  assignments = []

  # Concatenate the input channel data and extra signals to create the concat channel
  assignments.extend(generate_concat(
      in_channel_name, in_data_bitwidth, out_channel_name, concat_layout, array_size))

  # Forward the input channel handshake to the concat channel
  assignments.append(f"{out_channel_name}_valid <= {in_channel_name}_valid;")
  assignments.append(f"{in_channel_name}_ready <= {out_channel_name}_ready;")

  return assignments


def generate_slice_and_handshake(in_channel_name: str, out_channel_name: str, out_data_bitwidth: int, concat_layout: ConcatLayout, array_size=0) -> list[str]:
  """
  Generate VHDL assignments for slicing input channel data into output channel
  data, and forwarding the handshake signals.
  Args:
    in_channel_name (str): Name of the input channel.
    out_channel_name (str): Name of the output channel.
    out_data_bitwidth (int): Bitwidth of the output channel data.
    concat_layout (ConcatLayout): Layout for slicing extra signals.
    array_size (int): Size of the array if the channel is an array. Defaults to 0.
  Returns:
    assignments (list[str]): List of VHDL assignments for the slicing and handshake forwarding.
  """

  assignments = []

  # Slice the concat channel to create the output channel data and extra signals
  assignments.extend(generate_slice(
      in_channel_name, out_channel_name, out_data_bitwidth, concat_layout, array_size))

  # Forward the concat channel handshake to the output channel
  assignments.append(f"{out_channel_name}_valid <= {in_channel_name}_valid;")
  assignments.append(f"{in_channel_name}_ready <= {out_channel_name}_ready;")

  return assignments


def generate_mapping(channel: Channel, inner_channel_name: str) -> list[str]:
  """
  Generate VHDL port mappings of a channel, for the inner entity (port map (...)).
  Maps extra signals if present.
  `inner_channel_name` can differ from the name of `channel`. For example, the
  `channel` can be internally defined as `ins_inner`, while it still maps to the
  original channel name `ins` in the inner entity.
  Args:
    channel (Channel): Channel to generate mapping for.
    inner_channel_name (str): Name of the channel in the inner entity.
  Returns:
    mapping (list[str]): List of VHDL mapping strings for the channel.
  """

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
        f"{inner_channel_name}_{signal_name} => {channel_name}_{signal_name}")

  return mapping


def generate_default_mappings(channels: list[Channel]) -> str:
  """
  Generate VHDL mappings for the inner entity (port map (...)), using a list of
  channels where each channel maps to a channel of the same name without extra
  signals.
  """

  mappings = []
  for channel in channels:
    mappings.extend(generate_mapping({
        **channel,
        # Exclude extra signals from the mapping
        "extra_signals": {}
    }, channel["name"]))
  return ",\n      ".join(mappings)


def enumerate_channel_names(channels: list[Channel]) -> list[str]:
  """
  Enumerate channel names in the provided channel list.
  If a channel is an array named `ins`, include all of the individual channels
  (e.g., `ins_0`, `ins_1`, ...).
  """

  channel_names = []
  for channel in channels:
    size = channel.get("size", 0)

    if size == 0:
      channel_names.append(channel["name"])
    else:
      # Channel is an array
      for i in range(size):
        channel_names.append(f"{channel['name']}_{i}")

  return channel_names


def generate_signal_wise_forwarding(in_channel_names: list[str], out_channel_names: list[str], extra_signal_name: str) -> list[str]:
  """
  Generate VHDL assignments for forwarding extra signals from input
  channels to output channels.
  Args:
    in_channel_names (list[str]): List of input channel names.
    out_channel_names (list[str]): List of output channel names.
    extra_signal_name (str): Name of the extra signal to forward.
    extra_signal_bitwidth (int): Bitwidth of the extra signal.
  Returns:
    assignments (list[str]): List of VHDL assignments for the extra signal forwarding.
  """

  assignments = []

  # Collect all extra signals in the input channels
  in_extra_signal_names = []
  for in_channel_name in in_channel_names:
    signal_name = f"{in_channel_name}_{extra_signal_name}"
    in_extra_signal_names.append(signal_name)

  # Generate the forwarding expression for the extra signal
  expression = generate_forwarding_expression_for_signal(
      extra_signal_name, in_extra_signal_names)

  # Assign the expression to each output channel
  for out_channel_name in out_channel_names:
    signal_name = f"{out_channel_name}_{extra_signal_name}"
    assignments.append(f"{signal_name} <= {expression};")

  return assignments


def generate_extra_signal_decls(channel_name: str, extra_signals: ExtraSignals):
  """
  Generate VHDL declarations for extra signals in a channel.
  """

  return [
      generate_internal_signal_vector(
          f"{channel_name}_{signal_name}", signal_bitwidth)
      for signal_name, signal_bitwidth in extra_signals.items()
  ]


def generate_channel_decls(channel: Channel) -> list[str]:
  """
  Generate VHDL declarations for a channel.
  Args:
    channel (Channel): Channel to generate declarations for.
  Returns:
    decls (list[str]): List of VHDL declarations for the channel.
  """

  decls = []

  size = channel.get("size", 0)
  if size == 0:
    # Declare data signal if present
    if channel["bitwidth"] > 0:
      decls.append(generate_internal_signal_vector(
          channel["name"], channel["bitwidth"]))

    # Declare handshake signals
    decls.append(generate_internal_signal(
        f"{channel['name']}_valid"))
    decls.append(generate_internal_signal(
        f"{channel['name']}_ready"))

    # Declare extra signals if present
    extra_signals = channel.get("extra_signals", {})
    decls.extend(generate_extra_signal_decls(channel["name"], extra_signals))
  else:
    # Channel is an array

    # Declare data signal if present
    if channel["bitwidth"] > 0:
      decls.append(generate_internal_signal_array(
          channel["name"], channel["bitwidth"], size))

    # Declare handshake signals
    decls.append(generate_internal_signal_vector(
        f"{channel['name']}_valid", size))
    decls.append(generate_internal_signal_vector(
        f"{channel['name']}_ready", size))

    # Declare extra signals if present
    extra_signals = channel.get("extra_signals", {})
    decls.extend(generate_extra_signal_decls(channel["name"], extra_signals))

  return decls
