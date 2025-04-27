def generate_internal_signal(name: str) -> str:
  return f"signal {name} : std_logic;"


def generate_internal_signal_vector(name: str, bitwidth: int) -> str:
  return f"signal {name} : std_logic_vector({bitwidth - 1} downto 0);"


def generate_internal_signal_array(name: str, bitwidth: int, size: int) -> str:
  return f"signal {name} : data_array({size - 1} downto 0)({bitwidth - 1} downto 0);"
