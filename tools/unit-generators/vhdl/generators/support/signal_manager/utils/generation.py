from .types import Channel
from .concat import ConcatLayout
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
      - Example: When `in_channel_name` is `ins` and `in_data_bitwidth` is 32:
        ins_concat(32 - 1 downto 0) <= ins;
        ins_concat(32 downto 32) <= ins_spec;
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
      - Example: When `in_channel_name` is `outs_concat` and `out_channel_name` is `outs`:
        outs <= outs_concat(32 - 1 downto 0);
        outs_spec <= outs_concat(32 downto 32);
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
      - Example: When `in_channel_name` is `ins` and `out_channel_name` is `ins_concat`:
        ins_concat(32 - 1 downto 0) <= ins;
        ins_concat(32 downto 32) <= ins_spec;
        ins_concat_valid <= ins_valid;
        ins_ready <= ins_concat_ready;
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
      - Example: When `in_channel_name` is `outs_concat` and `out_channel_name` is `outs`:
        outs <= outs_concat(32 - 1 downto 0);
        outs_spec <= outs_concat(32 downto 32);
        outs_valid <= outs_concat_valid;
        outs_concat_ready <= outs_ready;
    """

    assignments = []

    # Slice the concat channel to create the output channel data and extra signals
    assignments.extend(generate_slice(
        in_channel_name, out_channel_name, out_data_bitwidth, concat_layout, array_size))

    # Forward the concat channel handshake to the output channel
    assignments.append(f"{out_channel_name}_valid <= {in_channel_name}_valid;")
    assignments.append(f"{in_channel_name}_ready <= {out_channel_name}_ready;")

    return assignments


def generate_mapping(original_channel_name: str, channel: Channel) -> list[str]:
    """
    Generate VHDL port mappings of a channel, for the inner entity (port map (...)).
    Maps extra signals if present.
    `original_channel_name` can differ from the name of `channel`. For example, the
    `channel` can be internally defined as `ins_inner`, while it still maps to the
    original channel name `ins` in the inner entity.
    Args:
      original_channel_name (str): Name of the channel in the inner entity.
      channel (Channel): Channel to generate mapping for.
    Returns:
      mapping (list[str]): List of VHDL mapping strings for the channel.
      - Example: When `original_channel_name` is `ins` and `channel` is named `ins_inner`,
        carrying `spec` extra signal:
        ins => ins_inner,
        ins_valid => ins_inner_valid,
        ins_ready => ins_inner_ready,
        ins_spec => ins_inner_spec
    """

    mapping: list[str] = []
    channel_name = channel["name"]
    channel_extra_signals = channel.get("extra_signals", {})
    channel_bitwidth = channel["bitwidth"]

    if channel_bitwidth > 0:
        # Mapping for data signal if present
        mapping.append(f"{original_channel_name} => {channel_name}")

    # Mapping for handshake signals
    mapping.append(f"{original_channel_name}_valid => {channel_name}_valid")
    mapping.append(f"{original_channel_name}_ready => {channel_name}_ready")

    for signal_name in channel_extra_signals:
        mapping.append(
            f"{original_channel_name}_{signal_name} => {channel_name}_{signal_name}")

    return mapping


def generate_default_mappings(channels: list[Channel]) -> str:
    """
    Generate VHDL mappings for the inner entity (port map (...)), using a list of
    channels where each channel maps to a channel of the same name without extra
    signals.
    - Example:
      ins => ins,
      ins_valid => ins_valid,
      ins_ready => ins_ready
    """

    mappings = []
    for channel in channels:
        # Map signals without extra signals to the inner entity.
        channel_without_extra_signals: Channel = \
            {**channel, "extra_signals": {}}

        mappings.extend(generate_mapping(
            channel["name"], channel_without_extra_signals))
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
      - Example: When `in_channel_names` is `["lhs", "rhs"]` and
       `out_channel_names` is `["trueResult", "falseResult"]`:
       trueResult_spec <= lhs_spec or rhs_spec;
       falseResult_spec <= lhs_spec or rhs_spec;
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
