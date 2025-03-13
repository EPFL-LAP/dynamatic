# See docs/Specs/SignalManager.md

from collections.abc import Callable
from generators.support.utils import get_default_extra_signal_value, ExtraSignalMapping


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]) -> str:
  debugging_info = f"-- signal manager generated info: {name}, {params}\n"

  in_ports = params["in_ports"]
  out_ports = params["out_ports"]
  type = params["type"]

  if type == "normal":
    extra_signals = params["extra_signals"]
    signal_manager = _generate_normal_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "buffered":
    extra_signals = params["extra_signals"]
    latency = params["latency"]
    signal_manager = _generate_buffered_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner, latency)
  elif type == "concat":
    extra_signals = params["extra_signals"]
    signal_manager = _generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "bbmerge":
    size = params["size"]
    data_in_name = params["data_in_name"]
    index_name = params["index_name"]
    out_extra_signals = params["out_extra_signals"]
    spec_inputs = params["spec_inputs"]
    signal_manager = _generate_bbmerge_signal_manager(
        name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, spec_inputs, generate_inner)
  else:
    raise ValueError(f"Unsupported signal manager type: {type}")

  return signal_manager + debugging_info


def generate_entity(entity_name, in_ports, out_ports) -> str:
  """
  Generate entity for signal manager, based on input and output ports
  """

  # Unify input and output ports, and add direction
  unified_ports = []
  for port in in_ports:
    unified_ports.append({
        **port,
        "direction": "in"
    })
  for port in out_ports:
    unified_ports.append({
        **port,
        "direction": "out"
    })

  port_decls = []
  # Add port declarations for each port
  for port in unified_ports:
    dir = port["direction"]
    ready_dir = "out" if dir == "in" else "in"

    name = port["name"]
    bitwidth = port["bitwidth"]
    extra_signals = port.get("extra_signals", {})
    port_2d = port.get("2d", False)

    if not port_2d:
      # Usual case

      # Generate data signal if present
      if bitwidth > 0:
        port_decls.append(
            f"    {name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

      port_decls.append(f"    {name}_valid : {dir} std_logic")
      port_decls.append(f"    {name}_ready : {ready_dir} std_logic")

      # Generate extra signals for this input port
      for signal_name, signal_bitwidth in extra_signals.items():
        port_decls.append(
            f"    {name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
    else:
      # Port is 2d
      size = port["size"]

      # Generate data_array signal declarations for 2d input port with bitwidth > 0
      if bitwidth > 0:
        port_decls.append(
            f"    {name} : {dir} data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0)")

      # Use std_logic_vector for valid/ready of 2d input port
      port_decls.append(
          f"    {name}_valid : {dir} std_logic_vector({size} - 1 downto 0)")
      port_decls.append(
          f"    {name}_ready : {ready_dir} std_logic_vector({size} - 1 downto 0)")

      # Use extra_signals_list if available to handle per-port extra signals
      use_extra_signals_list = "extra_signals_list" in port

      # Generate extra signal declarations for each item in the 2d input port
      for i in range(size):
        if use_extra_signals_list:
          current_extra_signals = port["extra_signals_list"][i]
        else:
          # Use the same extra signals for all items
          current_extra_signals = extra_signals

        # The netlist generator declares extra signals independently for each item,
        # in contrast to ready/valid signals.
        for signal_name, signal_bitwidth in current_extra_signals.items():
          port_decls.append(
              f"    {name}_{i}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")

  return f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity {entity_name} is
  port(
    clk : in std_logic;
    rst : in std_logic;
{";\n".join(port_decls)}
  );
end entity;
"""


def _forward_extra_signals(extra_signals: dict[str, int], in_ports) -> dict[str, str]:
  """
  Calculate how each extra signal is forwarded to the output ports.
  We assume that all extra signals are ORed currently.
  Result is a dict of extra signal names to VHDL expressions.
  e.g., {"spec": "lhs_spec or rhs_spec", "tag0": "lhs_tag0 or rhs_tag0"}
  If no inputs are provided, we use the default values.
  e.g., {"spec": "\"0\"", "tag0": "\"0\""}
  """

  forwarded_extra_signals: dict[str, str] = {}
  # Calculate forwarded extra signals
  for signal_name in extra_signals:
    in_extra_signals = []

    if not in_ports:
      # Use default values for extra signals
      forwarded_extra_signals[signal_name] = get_default_extra_signal_value(
          signal_name)
    else:
      # Collect extra signals from all input ports
      for in_port in in_ports:
        port_name = in_port["name"]
        in_extra_signals.append(f"{port_name}_{signal_name}")

      # OR all extra signals from input ports
      forwarded_extra_signals[signal_name] = f" or ".join(in_extra_signals)

  return forwarded_extra_signals


def generate_inner_port_forwarding(ports) -> str:
  """
  Generate port forwarding for inner entity
  e.g.,
      lhs => lhs,
      lhs_valid => lhs_valid,
      lhs_ready => lhs_ready
  """
  forwardings = []
  for port in ports:
    port_name = port["name"]
    bitwidth = port["bitwidth"]

    # Forward data if present
    if bitwidth > 0:
      forwardings.append(f"      {port_name} => {port_name}")

    forwardings.append(f"      {port_name}_valid => {port_name}_valid")
    forwardings.append(f"      {port_name}_ready => {port_name}_ready")

  return ",\n".join(forwardings)


def _generate_normal_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]) -> str:
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  forwarded_extra_signals = _forward_extra_signals(
      extra_signals, in_ports)

  # Assign all extra signals for each output port, based on forwarded_extra_signals.
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []
  for out_port in out_ports:
    port_name = out_port["name"]

    # Assign all extra signals to this output port
    for signal_name in extra_signals:
      extra_signal_assignments.append(
          f"  {port_name}_{signal_name} <= {forwarded_extra_signals[signal_name]};")

  inner_port_forwarding = generate_inner_port_forwarding(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
begin
  -- Forward extra signals to output ports
{"\n".join(extra_signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{inner_port_forwarding}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_buffered_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str], latency: int):
  # Delayed import to avoid circular dependency
  from generators.handshake.ofifo import generate_ofifo

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  forwarded_extra_signals = _forward_extra_signals(
      extra_signals, in_ports)

  # Construct extra signal mapping to concatenate extra signals
  extra_signal_mapping = ExtraSignalMapping(extra_signals)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate buffer to store (concatenated) extra signals
  buff_name = f"{name}_buff"
  buff = generate_ofifo(buff_name, {
      "num_slots": latency,
      "bitwidth": extra_signals_bitwidth
  })

  # Generate transfer logic
  first_in_port_name = in_ports[0]["name"]
  first_out_port_name = out_ports[0]["name"]
  transfer_logic = f"""
  transfer_in <= {first_in_port_name}_valid and {first_in_port_name}_ready;
  transfer_out <= {first_out_port_name}_valid and {first_out_port_name}_ready;
"""

  # Concat/split extra signals for buffer input/output.
  signal_assignments = []

  # Iterate over all extra signals
  for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
    # Concat extra signals for buffer input.
    signal_assignments.append(
        f"  buff_in({msb} downto {lsb}) <= {forwarded_extra_signals[signal_name]};")

    # Assign extra signals to all output ports
    for out_port in out_ports:
      port_name = out_port["name"]

      # Split extra signals from buffer output.
      signal_assignments.append(
          f"  {port_name}_{signal_name} <= buff_out({msb} downto {lsb});")

  forwarding = generate_inner_port_forwarding(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  signal buff_in, buff_out : std_logic_vector({extra_signals_bitwidth} - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
  -- Transfer signal assignments
{transfer_logic}

  -- Concat/split extra signals for buffer input/output
{"\n".join(signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{forwarding}
    );

  -- Generate ofifo to store extra signals
  -- num_slots = {latency}, bitwidth = {extra_signals_bitwidth}
  buff : entity work.{buff_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => buff_in,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => buff_out,
      outs_valid => open,
      outs_ready => transfer_out
    );
end architecture;
"""

  return inner + buff + entity + architecture


def generate_concat_signal_decls(ports, extra_signals_bitwidth, ignore=[]) -> str:
  """
  Declare signals for concatenated data and extra signals
  e.g., signal lhs_inner : std_logic_vector(33 - 1 downto 0); // 32 (data) + 1 (spec)
  """
  signal_decls = []
  for port in ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]
    port_2d = port.get("2d", False)

    # Ignore some ports
    if port_name in ignore:
      continue

    # Concatenated bitwidth
    full_bitwidth = extra_signals_bitwidth + port_bitwidth

    if port_2d:
      port_size = port["size"]

      # Inner signal is data_array
      signal_decls.append(
          f"  signal {port_name}_inner : data_array({port_size} - 1 downto 0)({full_bitwidth} - 1 downto 0);")
    else:
      signal_decls.append(
          f"  signal {port_name}_inner : std_logic_vector({full_bitwidth} - 1 downto 0);")

  return "\n".join(signal_decls)


def generate_concat_logic(in_ports, out_ports, extra_signal_mapping, ignore=[]):
  """
  Generate concat logic for all input/output ports
  e.g.,
  lhs_inner(31 downto 0) <= lhs;
  lhs_inner(32 downto 32) <= lhs_spec;
  ...
  result <= result_inner(31 downto 0);
  result_spec <= result_inner(32 downto 32);
  """
  concat_logic = []
  for port in in_ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]
    port_2d = port.get("2d", False)

    # Ignore some ports
    if port_name in ignore:
      continue

    if port_2d:
      port_size = port["size"]
      for i in range(port_size):
        # Include data if present
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}_inner({i})({port_bitwidth} - 1 downto 0) <= {port_name}({i});")

        # Include all extra signals
        for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
          concat_logic.append(
              f"  {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{i}_{signal_name};")
    else:
      # Include data if present
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name}_inner({port_bitwidth} - 1 downto 0) <= {port_name};")

      # Include all extra signals
      for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
        concat_logic.append(
            f"  {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{signal_name};")

  for port in out_ports:
    port_name = port["name"]
    port_bitwidth = port["bitwidth"]
    port_2d = port.get("2d", False)

    # Ignore some ports
    if port_name in ignore:
      continue

    if port_2d:
      port_size = port["size"]
      for i in range(port_size):
        # Extract data if present
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}({i}) <= {port_name}_inner({i})({port_bitwidth} - 1 downto 0);")

        # Extract all extra signals
        for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
          concat_logic.append(
              f"  {port_name}_{i}_{signal_name} <= {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth});")
    else:
      # Extract data if present
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name} <= {port_name}_inner({port_bitwidth} - 1 downto 0);")

      # Extract all extra signals
      for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
        concat_logic.append(
            f"  {port_name}_{signal_name} <= {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth});")

  return "\n".join(concat_logic)


def _generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Construct extra signal mapping to concatenate extra signals
  extra_signal_mapping = ExtraSignalMapping(extra_signals)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_logic(
      in_ports, out_ports, extra_signal_mapping)

  # Port forwarding for the inner entity
  # We can't use _generate_inner_port_forwarding() because:
  # (1) Data is always forwarded, regardless of (port's original) bitwidth, due to the concatenation.
  # (2) Data ports must be renamed to `_inner`.
  forwardings = []
  for port in in_ports + out_ports:
    port_name = port["name"]

    forwardings.append(f"      {port_name} => {port_name}_inner")
    forwardings.append(f"      {port_name}_valid => {port_name}_valid")
    forwardings.append(f"      {port_name}_ready => {port_name}_ready")

  architecture = f"""
-- Architecture of signal manager (concat)
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
{",\n".join(forwardings)}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_bbmerge_signal_manager(name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, spec_inputs, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Construct extra signal mapping to concatenate extra signals
  extra_signal_mapping = ExtraSignalMapping(out_extra_signals)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare and assign default spec bits for inputs without them
  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"  signal {data_in_name}_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  lacking_spec_port_assignments = [
      f"  {data_in_name}_{i}_spec <= {get_default_extra_signal_value("spec")};" for i in lacking_spec_ports
  ]

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports, extra_signals_bitwidth, ignore=[index_name])

  # Assign inner concatenated signals
  concat_logic = generate_concat_logic(
      in_ports, out_ports, extra_signal_mapping, ignore=[index_name])

  # Port forwarding for the inner entity
  # We can't use _generate_inner_port_forwarding() because:
  # (1) Data is always forwarded, regardless of (port's original) bitwidth, due to the concatenation.
  # (2) Data ports must be renamed to `_inner`.
  forwardings = []
  for port in in_ports + out_ports:
    port_name = port["name"]

    # Forward the original data signal for the index port
    if port_name == index_name:
      forwardings.append(f"      {port_name} => {port_name}")
    else:
      forwardings.append(f"      {port_name} => {port_name}_inner")
    forwardings.append(f"      {port_name}_valid => {port_name}_valid")
    forwardings.append(f"      {port_name}_ready => {port_name}_ready")

  architecture = f"""
-- Architecture of signal manager (bbmerge)
architecture arch of {name} is
  -- Lacking spec inputs
{"\n".join(lacking_spec_port_decls)}
  -- Concatenated data and extra signals
{concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
{"\n".join(lacking_spec_port_assignments)}

  -- Concatenate data and extra signals
{concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{",\n".join(forwardings)}
    );
end architecture;
"""

  return inner + entity + architecture
