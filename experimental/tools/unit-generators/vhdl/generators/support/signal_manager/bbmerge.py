from collections.abc import Callable
from .utils.entity import generate_entity
from .utils.forwarding import get_default_extra_signal_value
from .utils.concat import generate_concat_signal_decls_from_ports, ConcatLayout, generate_concat_port_assignments_from_ports
from .utils.mapping import generate_inner_port_mapping, generate_concat_mappings
from .utils.types import Port, ExtraSignals


def _generate_bbmerge_lacking_spec_statements(spec_inputs: list[int], size: int, data_in_name: str) -> tuple[str, str]:
  """
  Generate declarations and default assignments for `spec` signals
  on input ports that don't explicitly carry them.

  Returns:
    - Tuple of (declarations, assignments) as strings.

  Example:
    - decls: signal lhs_0_spec : std_logic_vector(0 downto 0);
    - assigns: lhs_0_spec <= "0";
  """

  lacking_spec_ports = [
      i for i in range(size) if i not in spec_inputs
  ]
  lacking_spec_port_decls = [
      f"signal {data_in_name}_{i}_spec : std_logic_vector(0 downto 0);" for i in lacking_spec_ports
  ]
  lacking_spec_port_assignments = [
      f"{data_in_name}_{i}_spec <= {get_default_extra_signal_value("spec")};" for i in lacking_spec_ports
  ]
  return "\n  ".join(lacking_spec_port_decls), "\n  ".join(lacking_spec_port_assignments).lstrip()


def generate_mux_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
    size: int,
    data_in_name: str,
    index_name: str,
    out_extra_signals: ExtraSignals,
    spec_inputs: list[int],
    generate_inner: Callable[[str], str]
) -> str:
  """
  Generate a signal manager architecture for a mux.

  This handles extra signal concatenation, default spec assignment,
  and port mapping for inner mux entities.

  Args:
    name: The name of the signal manager entity.
    in_ports: List of input ports including data ports and index port.
    out_ports: List of output ports.
    size: The number of data input ports.
    data_in_name: The name of the data input port (e.g., "ins").
    index_name: The name of the index port (e.g., "index").
    out_extra_signals: Dictionary of extra signals on the data output port.
    spec_inputs: List of indices of data inputs that have a "spec" signal.
    generate_inner: A function to generate the inner entity.

  Returns:
    A string representing the complete VHDL architecture for the signal manager.
  """
  entity = generate_entity(name, in_ports, out_ports)

  in_ports_without_index = [
      port for port in in_ports if port["name"] != index_name]
  index_port = [
      port for port in in_ports if port["name"] == index_name][0]

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(out_extra_signals)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Generate default `spec` bits for inputs that lack them
  lacking_spec_port_decls, lacking_spec_port_assignments = _generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, data_in_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = "\n  ".join(generate_concat_signal_decls_from_ports(
      in_ports_without_index + out_ports, extra_signals_bitwidth))

  # Assign inner concatenated signals
  concat_logic = "\n  ".join(generate_concat_port_assignments_from_ports(
      in_ports_without_index, out_ports, concat_layout))

  # Map all ports to inner entity:
  #   - Forward concatenated extra signal vectors
  #   - Pass through index port as-is
  mappings = "\n      ".join(generate_concat_mappings(
      in_ports_without_index + out_ports, extra_signals_bitwidth) +
      generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (mux)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

  -- Concatenate data and extra signals
  {concat_logic}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture


def _generate_cmerge_index_extra_signal_assignments(index_name: str, index_extra_signals: ExtraSignals) -> str:
  """
  Generate VHDL assignments for extra signals on the index port (cmerge).

  Example:
    - index_tag0 <= "0";
  """

  # TODO: Extra signals on the index port are not tested
  index_extra_signals_list = []
  for signal_name in index_extra_signals:
    index_extra_signals_list.append(
        f"  {index_name}_{signal_name} <= {get_default_extra_signal_value(signal_name)};")
  return "\n  ".join(index_extra_signals_list)


def generate_cmerge_signal_manager(
    name: str,
    in_ports: list[Port],
    out_ports: list[Port],
    size: int,
    data_in_name: str,
    index_name: str,
    out_extra_signals: ExtraSignals,
    index_extra_signals: ExtraSignals,
    spec_inputs: list[int],
    generate_inner: Callable[[str], str]
) -> str:
  """
  Generate a signal manager architecture for a cmerge.

  Similar to the mux version, but assigns index extra signals.

  Args:
    name: The name of the signal manager entity.
    in_ports: List of input ports.
    out_ports: List of output ports including data port and index port.
    size: The number of data input ports.
    data_in_name: The name of the data input port (e.g., "ins").
    index_name: The name of the index port (e.g., "index").
    out_extra_signals: Dictionary of extra signals on the data output port.
    index_extra_signals: Dictionary of extra signals on the index port.
    spec_inputs: List of indices of data inputs that have a "spec" signal.
    generate_inner: A function to generate the inner entity.

  Returns:
    A string representing the complete VHDL architecture for the signal manager.
  """
  entity = generate_entity(name, in_ports, out_ports)

  out_ports_without_index = [
      port for port in out_ports if port["name"] != index_name]
  index_port = [
      port for port in out_ports if port["name"] == index_name][0]

  # Layout info for how extra signals are packed into one std_logic_vector
  concat_layout = ConcatLayout(out_extra_signals)
  extra_signals_bitwidth = concat_layout.total_bitwidth

  inner_name = f"{name}_inner"
  inner = generate_inner(inner_name)

  # Generate default `spec` bits for inputs that lack them
  lacking_spec_port_decls, lacking_spec_port_assignments = _generate_bbmerge_lacking_spec_statements(
      spec_inputs, size, data_in_name)

  # Declare inner concatenated signals for all input/output ports
  concat_signal_decls = "\n  ".join(generate_concat_signal_decls_from_ports(
      in_ports + out_ports_without_index, extra_signals_bitwidth))

  # Assign inner concatenated signals
  concat_logic = "\n  ".join(generate_concat_port_assignments_from_ports(
      in_ports, out_ports_without_index, concat_layout))

  # Assign index extra signals
  index_extra_signal_assignments = _generate_cmerge_index_extra_signal_assignments(
      index_name, index_extra_signals)

  # Map all ports to inner entity:
  #   - Forward concatenated extra signal vectors
  #   - Pass through index port as-is
  mappings = "\n      ".join(generate_concat_mappings(
      in_ports + out_ports_without_index, extra_signals_bitwidth) +
      generate_inner_port_mapping(index_port))

  architecture = f"""
-- Architecture of signal manager (cmerge)
architecture arch of {name} is
  -- Lacking spec inputs
  {lacking_spec_port_decls}
  -- Concatenated data and extra signals
  {concat_signal_decls}
begin
  -- Assign default spec bit values if not provided
  {lacking_spec_port_assignments}

  -- Concatenate data and extra signals
  {concat_logic}

  -- Assign index extra signals (if any)
  {index_extra_signal_assignments}

  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      {mappings}
    );
end architecture;
"""

  return inner + entity + architecture
