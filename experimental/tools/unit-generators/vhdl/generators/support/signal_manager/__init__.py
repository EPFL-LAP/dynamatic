# See docs/Specs/SignalManager.md

from collections.abc import Callable
from .entity import generate_entity
from .forwarding import forward_extra_signals, get_default_extra_signal_value, generate_forwarding_assignments
from .types import Port, ExtraSignals
from .mapping import generate_simple_inner_port_mappings, generate_inner_port_mapping
from .concat import generate_concat_signal_decls, ConcatenationInfo, generate_concat_port_assignments


def generate_signal_manager(name, params, generate_inner: Callable[[str], str]) -> str:
  debugging_info = f"-- Signal manager generation info: {name}, {params}\n"

  in_ports: list[Port] = params["in_ports"]
  out_ports: list[Port] = params["out_ports"]
  type = params["type"]

  if type == "normal":
    extra_signals: ExtraSignals = params["extra_signals"]
    signal_manager = _generate_normal_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner)
  elif type == "buffered":
    extra_signals = params["extra_signals"]
    latency = params["latency"]
    signal_manager = _generate_buffered_signal_manager(
        name, in_ports, out_ports, extra_signals, generate_inner, latency)
  elif type == "concat":
    extra_signals = params["extra_signals"]
    ignore_ports = params.get("ignore_ports", [])
    signal_manager = _generate_concat_signal_manager(
        name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner)
  elif type == "bbmerge":
    size = params["size"]
    data_in_name = params["data_in_name"]
    index_name = params["index_name"]
    out_extra_signals = params["out_extra_signals"]
    index_extra_signals = params["index_extra_signals"]
    index_dir = params["index_dir"]
    spec_inputs = params["spec_inputs"]
    signal_manager = _generate_bbmerge_signal_manager(
        name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, index_extra_signals, index_dir, spec_inputs, generate_inner)
  else:
    raise ValueError(f"Unsupported signal manager type: {type}")

  return signal_manager + debugging_info


def _generate_normal_signal_assignments(in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals) -> str:
  """
  e.g., result_spec <= lhs_spec or rhs_spec;
  """
  return "\n".join(generate_forwarding_assignments(
      in_port_names=[port["name"] for port in in_ports],
      out_port_names=[port["name"] for port in out_ports],
      extra_signal_names=list(extra_signals)
  )).lstrip()


def _generate_normal_signal_manager(name: str, in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals, generate_inner: Callable[[str], str]) -> str:
  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  entity = generate_entity(name, in_ports, out_ports)

  extra_signal_assignments = _generate_normal_signal_assignments(
      in_ports, out_ports, extra_signals)

  mappings = generate_simple_inner_port_mappings(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (normal)
architecture arch of {name} is
begin
  -- Forward extra signals to output ports
  {extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_buffered_transfer_logic(in_ports: list[Port], out_ports: list[Port]):
  first_in_port_name = in_ports[0]["name"]
  first_out_port_name = out_ports[0]["name"]

  return f"""
  transfer_in <= {first_in_port_name}_valid and {first_in_port_name}_ready;
  transfer_out <= {first_out_port_name}_valid and {first_out_port_name}_ready;""".lstrip()


def _generate_buffered_signal_assignments(in_ports: list[Port], out_ports: list[Port], concat_info: ConcatenationInfo, extra_signals: ExtraSignals) -> str:
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
        f"  buff_in({msb} downto {lsb}) <= {forwarded_extra_signals[signal_name]};")

    # Assign extra signals to all output ports
    for out_port in out_ports:
      port_name = out_port["name"]

      # Split extra signals from buffer output.
      signal_assignments.append(
          f"  {port_name}_{signal_name} <= buff_out({msb} downto {lsb});")

  return "\n".join(signal_assignments).lstrip()


def _generate_buffered_signal_manager(name: str, in_ports: list[Port], out_ports: list[Port], extra_signals: ExtraSignals, generate_inner: Callable[[str], str], latency: int):
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
  transfer_logic = _generate_buffered_transfer_logic(in_ports, out_ports)

  signal_assignments = _generate_buffered_signal_assignments(
      in_ports, out_ports, concat_info, extra_signals)

  mappings = generate_simple_inner_port_mappings(in_ports + out_ports)

  architecture = f"""
-- Architecture of signal manager (buffered)
architecture arch of {name} is
  signal buff_in, buff_out : std_logic_vector({extra_signals_bitwidth} - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
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


def generate_concat_mappings(ports: list[Port], extra_signals_bitwidth: int, handled_extra_signals: ExtraSignals, ignore_ports: list[str]) -> str:
  mappings = []
  for port in ports:
    port_name = port["name"]
    port_extra_signals = port.get("extra_signals", {})

    mapping_port = port.copy()
    mapping_port["bitwidth"] += extra_signals_bitwidth

    mapping_port_name = port_name if port_name in ignore_ports else f"{port_name}_inner"

    mapping_extra_signals = [signal_name
                             for signal_name in port_extra_signals
                             if signal_name not in handled_extra_signals]

    mappings += generate_inner_port_mapping(
        mapping_port, mapping_port_name, mapping_extra_signals)

  return ",\n".join(mappings).lstrip()


def _generate_concat_signal_manager(name, in_ports, out_ports, extra_signals, ignore_ports, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Exclude specified ports for concatenation
  filtered_in_ports = [
      port for port in in_ports if not port["name"] in ignore_ports]
  filtered_out_ports = [
      port for port in out_ports if not port["name"] in ignore_ports]

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      filtered_in_ports + filtered_out_ports, extra_signals_bitwidth)

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments(
      filtered_in_ports, filtered_out_ports, concat_info)

  # Port forwarding for the inner entity
  forwardings = generate_concat_mappings(
      in_ports + out_ports, extra_signals_bitwidth, extra_signals, ignore_ports)

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
      {forwardings}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_bbmerge_lacking_spec_statements(spec_inputs, size, data_in_name):
  """
  e.g.,
  - decls: signal lhs_0_spec : std_logic_vector(0 downto 0);
  - assigns: lhs_0_spec <= "0";
  """
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
  return "\n".join(lacking_spec_port_decls).lstrip(), "\n".join(lacking_spec_port_assignments).lstrip()


def _generate_bbmerge_index_extra_signal_assignments(index_name, index_extra_signals, index_dir) -> str:
  """
  e.g., index_tag0 <= "0";
  """
  # TODO: Extra signals for index port are not tested
  if index_dir == "out" and index_extra_signals:
    index_extra_signals_list = []
    for signal_name in index_extra_signals:
      index_extra_signals_list.append(
          f"  {index_name}_{signal_name} <= {get_default_extra_signal_value(signal_name)};")
    return "\n".join(index_extra_signals_list)
  return ""


def _generate_bbmerge_signal_assignments(lacking_spec_port_assignments, concat_logic, index_extra_signal_assignments) -> str:
  template = f"""
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

  -- Concatenate data and extra signals
  {concat_logic}
"""

  if index_extra_signal_assignments:
    template += f"""
  -- Assign index extra signals
  {index_extra_signal_assignments}
"""

  return template.lstrip()


def _generate_bbmerge_signal_manager(name, in_ports, out_ports, size, data_in_name, index_name, out_extra_signals, index_extra_signals, index_dir, spec_inputs, generate_inner: Callable[[str], str]):
  entity = generate_entity(name, in_ports, out_ports)

  # Get concatenation details for extra signals
  concat_info = ConcatenationInfo(out_extra_signals)
  extra_signals_bitwidth = concat_info.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  lacking_spec_port_decls, lacking_spec_port_assignments = _generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, data_in_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = generate_concat_signal_decls(
      in_ports + out_ports, extra_signals_bitwidth, ignore=[index_name])

  # Assign inner concatenated signals
  concat_logic = generate_concat_port_assignments(
      in_ports, out_ports, concat_info, ignore=[index_name])

  # Assign index extra signals
  index_extra_signal_assignments = _generate_bbmerge_index_extra_signal_assignments(
      index_name, index_extra_signals, index_dir)

  signal_assignments = _generate_bbmerge_signal_assignments(
      lacking_spec_port_assignments, concat_logic, index_extra_signal_assignments)

  # Port forwarding for the inner entity
  forwardings = generate_concat_mappings(
      in_ports + out_ports, extra_signals_bitwidth, out_extra_signals, [index_name])

  architecture = f"""
-- Architecture of signal manager (bbmerge)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  {signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {forwardings}
    );
end architecture;
"""

  return inner + entity + architecture
