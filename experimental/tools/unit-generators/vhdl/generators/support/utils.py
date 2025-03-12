def generate_extra_signal_ports(ports: list[tuple[str, str]], extra_signals: dict[str, int]) -> str:
  if not extra_signals:
    return ""
  return "    -- extra signal ports\n" + "\n".join([
      "\n".join([
          f"    {port}_{name} : {inout} std_logic_vector({bitwidth - 1} downto 0);"
          for name, bitwidth in extra_signals.items()
      ])
      for port, inout in ports
  ])

# For concat-type signal managers (e.g., tehb, fork)


class ExtraSignalMapping:
  # List of tuples of (extra_signal_name, (msb, lsb))
  mapping: list[tuple[str, tuple[int, int]]]
  total_bitwidth: int

  def __init__(self, offset: int = 0):
    """
    offset: The starting bitwidth of the extra signals (if data is present).
    """
    self.mapping = []
    self.total_bitwidth = offset

  def add(self, name: str, bitwidth: int):
    self.mapping.append(
        (name, (self.total_bitwidth + bitwidth - 1, self.total_bitwidth)))
    self.total_bitwidth += bitwidth

  def has(self, name: str) -> bool:
    return name in [name for name, _ in self.mapping]

  def get(self, name: str):
    return self.mapping[[name for name, _ in self.mapping].index(name)]


def generate_ins_concat_statements(in_name: str, in_inner_name: str, extra_signal_mapping: ExtraSignalMapping, bitwidth: int, indent=2, custom_data_name=None) -> str:
  """
  Generates the input signal concatenation statement.
  in_name: The name of the input signal. (e.g., "ins")
  in_inner_name: The name of the inner input signal. (e.g., "ins_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  bitwidth: The bitwidth of the data signal.
  e.g., ins_inner(31 downto 0) <= ins;
  ins_inner(32 downto 32) <= ins_spec;
  ins_inner(40 downto 33) <= ins_tag;
  """
  indent_str = " " * indent
  if custom_data_name is None:
    custom_data_name = in_name
  return f"{indent_str}{in_inner_name}({bitwidth - 1} downto 0) <= {custom_data_name};\n" + \
      generate_ins_concat_statements_dataless(
      in_name, in_inner_name, extra_signal_mapping, indent)


def generate_ins_concat_statements_dataless(in_name: str, in_inner_name: str, extra_signal_mapping: ExtraSignalMapping, indent=2) -> str:
  """
  Generates the input signal concatenation statement.
  in_name: The name of the input signal. (e.g., "ins")
  in_inner_name: The name of the inner input signal. (e.g., "ins_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  e.g., ins_inner(0 downto 0) <= ins_spec;
  ins_inner(8 downto 1) <= ins_tag;
  """
  indent_str = " " * indent
  return "\n".join([
      f"{indent_str}{in_inner_name}({msb} downto {lsb}) <= {in_name}_{name};" for name, (msb, lsb) in extra_signal_mapping.mapping
  ]) + "\n"


def generate_outs_concat_statements(out_name: str, out_inner_name: str, extra_signal_mapping: ExtraSignalMapping, bitwidth: int, indent=2, custom_data_name=None) -> str:
  """
  Generates the output signal concatenation statement.
  out_name: The name of the output signal. (e.g., "outs")
  out_inner_name: The name of the inner output signal. (e.g., "outs_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  bitwidth: The bitwidth of the data signal.
  e.g., outs <= outs_inner(31 downto 0)
  outs_spec <= outs_inner(32 downto 32)
  outs_tag <= outs_inner(40 downto 33)
  """
  indent_str = " " * indent
  if custom_data_name is None:
    custom_data_name = out_name
  return f"{indent_str}{custom_data_name} <= {out_inner_name}({bitwidth - 1} downto 0);\n" + \
      generate_outs_concat_statements_dataless(
      out_name, out_inner_name, extra_signal_mapping, indent)


def generate_outs_concat_statements_dataless(out_name: str, out_inner_name: str, extra_signal_mapping: ExtraSignalMapping, indent=2) -> str:
  """
  Generates the output signal concatenation statement.
  out_name: The name of the output signal. (e.g., "outs")
  out_inner_name: The name of the inner output signal. (e.g., "outs_inner")
  extra_signal_mapping: An ExtraSignalMapping object.
  bitwidth: The bitwidth of the data signal.
  e.g., outs_spec <= outs_inner(0 downto 0)
  outs_tag <= outs_inner(8 downto 1)
  """
  indent_str = " " * indent
  return "\n".join([
      f"{indent_str}{out_name}_{name} <= {out_inner_name}({msb} downto {lsb});" for name, (msb, lsb) in extra_signal_mapping.mapping
  ]) + "\n"


extra_signal_default_values = {
    "spec": "\"0\"",
}


def get_concat_extra_signals_bitwidth(extra_signals: dict[str, int]):
  return sum(extra_signals.values())
