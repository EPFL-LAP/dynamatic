from .forwarding import get_default_extra_signal_value


# Used by mux and control_merge signal managers
def generate_bbmerge_lacking_spec_statements(spec_inputs: list[int], size: int, data_in_name: str) -> tuple[str, str]:
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
