from collections.abc import Callable
from generators.support.utils import get_default_extra_signal_value, ExtraSignalMapping


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]):
  in_ports = params["in_ports"]
  out_ports = params["out_ports"]
  type = params["type"]

  if type == "normal":
    extra_signals = params["extra_signals"]
    return _generate_normal_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "buffered":
    extra_signals = params["extra_signals"]
    latency = params["latency"]
    return _generate_buffered_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner, latency)
  elif type == "concat":
    extra_signals = params["extra_signals"]
    return _generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "bbmerge":
    size = params["size"]
    data_in_name = params["data_in_name"]
    index_name = params["index_name"]
    out_extra_signals = params["out_extra_signals"]
    spec_inputs = params["spec_inputs"]
    return _generate_bbmerge_signal_manager(
        name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, spec_inputs, generate_inner)

  raise ValueError(f"Unsupported signal manager type: {type}")


def generate_entity(entity_name, in_ports, out_ports):
  # Unify input and output ports, and add direction
  unified_ports = []
  for port in in_ports:
    unified_ports.append({
        **port,
        "direction": "in"
    })
  for out_port in out_ports:
    unified_ports.append({
        **out_port,
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
      # Generate data signal port if present
      if bitwidth > 0:
        port_decls.append(
            f"    {name} : {dir} std_logic_vector({bitwidth} - 1 downto 0)")

      port_decls.append(f"    {name}_valid : {dir} std_logic")
      port_decls.append(f"    {name}_ready : {ready_dir} std_logic")

      # Generate extra signal port declarations for this input port
      for signal_name, signal_bitwidth in extra_signals.items():
        port_decls.append(
            f"    {name}_{signal_name} : {dir} std_logic_vector({signal_bitwidth} - 1 downto 0)")
    else:
      # Port is 2d
      size = port["size"]

      # Generate data_array port declarations for 2d input port with bitwidth > 0
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

      # Generate extra signal port declarations for each item in the 2d input port
      for i in range(size):
        if use_extra_signals_list:
          current_extra_signals = port["extra_signals_list"][i]
        else:
          current_extra_signals = extra_signals

        # The netlist generator declares extra signals independently per index,
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


def _calc_forwarded_extra_signals(extra_signals: dict[str, int], in_ports):
  """
  Calculate how each extra signal is forwarded to the output ports.
  We assume that all extra signals are ORed currently.
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


def _generate_inner_port_forwarding(ports):
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


def _generate_normal_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]):
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  extra_signal_exps = _calc_forwarded_extra_signals(
      extra_signals, in_ports)

  # Generate extra signal assignments for each output port and extra signal,
  # based on the extra signal expressions
  # e.g., result_spec <= lhs_spec or rhs_spec;
  extra_signal_assignments = []
  # Assign extra signals to all output ports
  for out_port in out_ports:
    port_name = out_port["name"]

    # Assign all extra signals to this output port
    for signal_name in extra_signals:
      extra_signal_assignments.append(
          f"  {port_name}_{signal_name} <= {extra_signal_exps[signal_name]};")

  forwarding = _generate_inner_port_forwarding(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
begin

{"\n".join(extra_signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{forwarding}
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

  extra_signal_exps = _calc_forwarded_extra_signals(
      extra_signals, in_ports)

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  # Generate buffer to store extra signals
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

  # Assign signals to concat/split extra signals for buffer input/output.
  signal_assignments = []

  # Iterate over all extra signals
  for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
    signal_assignments.append(
        f"  buff_in({msb} downto {lsb}) <= {extra_signal_exps[signal_name]};")

    # Assign extra signals to all output ports
    for out_port in out_ports:
      port_name = out_port["name"]

      signal_assignments.append(
          f"  {port_name}_{signal_name} <= buff_out({msb} downto {lsb});")

  forwarding = _generate_inner_port_forwarding(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
  signal buff_in, buff_out : std_logic_vector({extra_signals_bitwidth} - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
{transfer_logic}

{"\n".join(signal_assignments)}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
{forwarding}
    );

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


def generate_concat_signal_decls(ports, extra_signals_bitwidth, ignore=[]):
  """
  Declare signals for concatenated data and extra signals
  e.g., signal lhs_inner : std_logic_vector(32 downto 0); // 32 (data) + 1 (spec)
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
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}_inner({i})({port_bitwidth} - 1 downto 0) <= {port_name}({i});")

        for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
          concat_logic.append(
              f"  {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth}) <= {port_name}_{i}_{signal_name};")
    else:
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name}_inner({port_bitwidth} - 1 downto 0) <= {port_name};")

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
        if port_bitwidth > 0:
          concat_logic.append(
              f"  {port_name}({i}) <= {port_name}_inner({i})({port_bitwidth} - 1 downto 0);")

        for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
          concat_logic.append(
              f"  {port_name}_{i}_{signal_name} <= {port_name}_inner({i})({msb + port_bitwidth} downto {lsb + port_bitwidth});")
    else:
      if port_bitwidth > 0:
        concat_logic.append(
            f"  {port_name} <= {port_name}_inner({port_bitwidth} - 1 downto 0);")

      for signal_name, (msb, lsb) in extra_signal_mapping.mapping:
        concat_logic.append(
            f"  {port_name}_{signal_name} <= {port_name}_inner({msb + port_bitwidth} downto {lsb + port_bitwidth});")

  return "\n".join(concat_logic)


def _generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports, extra_signals_bitwidth)

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

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in out_extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  extra_signals_bitwidth = extra_signal_mapping.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare and assign spec bits for inputs without them
  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"  signal {data_in_name}_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  lacking_spec_port_assignments = [
      f"  {data_in_name}_{i}_spec <= {get_default_extra_signal_value("spec")};" for i in lacking_spec_ports
  ]

  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports, extra_signals_bitwidth, ignore=[index_name])

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
-- Architecture of signal manager (mergebb)
architecture arch of {name} is
  -- Lacking spec inputs
{"\n".join(lacking_spec_port_decls)}
  -- Concatenated data and extra signals
{concat_signal_decls}
begin
{"\n".join(lacking_spec_port_assignments)}
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
