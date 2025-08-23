from enum import Enum

class SignalSize():
    def __init__(self, bitwidth, number):
        self.bitwidth = bitwidth
        self.number = number

def signalSizeToTypeDeclaration(signal_size):
    if signal_size.bitwidth == 1:
        return "std_logic"
    else:
        return f"std_logic_vector({signal_size.bitwidth} - 1 downto 0)"

class EntitySignalType(Enum):
    INPUT = 1
    OUTPUT = 2


def makeEntitySignal(base_name, signal_size, entity_signal_type):
    match entity_signal_type:
        case EntitySignalType.INPUT:
            io_suffix = "i"
            direction = "in"
        case EntitySignalType.OUTPUT:
            io_suffix = "o"
            direction = "out"

    type_declaration = signalSizeToTypeDeclaration(signal_size)

    return f"""
  {base_name}_{io_suffix} : {direction} {type_declaration};
""".removeprefix("\n")

class Entity():
  def __init__(self):
    self.signals = ""

  def addInputSignal(self, signal_base_name, signal_size):
    self._addSignal(signal_base_name, signal_size, entity_signal_type=EntitySignalType.INPUT)

  def addOutputSignal(self, signal_base_name, signal_size):
    self._addSignal(signal_base_name, signal_size, entity_signal_type=EntitySignalType.OUTPUT)

  def _addSignal(
      self,
      signal_base_name,
      signal_size,
      entity_signal_type
  ):
    if signal_size.number == 1:
      newSignal = makeEntitySignal(
        signal_base_name,
        signal_size,
        entity_signal_type
      )
      self.signalss += newSignal
    else:
      for i in range(signal_size.number):
        newSignal = makeEntitySignal(
          f"{signal_base_name}_{i}",
          signal_size,
          entity_signal_type,
          )
        self.signals += newSignal

  def get(self, name, entity_type):
    self.signals = self.signals.lstrip()[:-1]
    entity = f"""
-- {entity_type}
entity {name} is
ports(
  rst : in std_logic;
  clk : in std_logic;
  {self.signals}
)
"""
    print(entity)
