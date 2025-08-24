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


def makeEntitySignal(base_name, signal):
    match signal.direction:
        case EntitySignalType.INPUT:
            io_suffix = "i"
            # with space at the end to match witdth of out
            direction = "in "
        case EntitySignalType.OUTPUT:
            io_suffix = "o"
            direction = "out"

    type_declaration = signalSizeToTypeDeclaration(signal.signal_size)

    name = f"{base_name}_{io_suffix}".ljust(20)

    full_declaration = f"{name} : {direction} {type_declaration};"
    if signal.signal_size.bitwidth == 0:
        full_declaration = f"-- {full_declaration}"
    return f"""
    {full_declaration}
""".removeprefix("\n")
    


def makeInstantiationSignal(base_name, signal):
    name = f"{base_name}"

    full_declaration = f"{name},"
    if signal.signal_size.bitwidth == 0:
        full_declaration = f"-- {full_declaration}"
    return f"""
      {full_declaration}
""".removeprefix("\n")
    

class Entity():
  def __init__(self):
    self.entity_signals = ""

  def __init__(self, declaration):
    self.entity_signals = ""
    self.instantiate_signals = ""
    for signal in declaration.io_signals:
        self._addSignal(signal)

  # def addInputSignal(self, signal_base_name, signal_size):
  #   self._addSignal(signal_base_name, signal_size, entity_signal_type=EntitySignalType.INPUT)


  # def addOutputSignal(self, signal_base_name, signal_size):
  #   self._addSignal(signal_base_name, signal_size, entity_signal_type=EntitySignalType.OUTPUT)


  def _addSignal(self, signal):
    if signal.comment is not None:
      self.entity_signals += signal.comment
      self.instantiate_signals += signal.comment
      
    if signal.signal_size.number == 1:
      newSignal = makeEntitySignal(
         signal.rtl_name,
         signal
        )
      self.entity_signals += newSignal

      newSignal = makeInstantiationSignal(
        signal.rtl_name,
         signal
      )
      self.instantiate_signals += newSignal

    else:
      for i in range(signal.signal_size.number):
        newSignal = makeEntitySignal(
          f"{signal.rtl_name}_{i}",
          signal
          )
        self.entity_signals += newSignal

        newSignal = makeInstantiationSignal(
          f"{signal.rtl_name}_{i}",
          signal
        )
        self.instantiate_signals += newSignal


  def get(self, name, entity_type):
    # remove leading whitespace
    # the required leading whitespace is present in the string
    # and remove final character, which is a semi-colon
    self.entity_signals = self.entity_signals.lstrip()[:-1]

    entity = f"""
-- {entity_type}
entity {name} is
  port(
    {self.entity_signals}
  );
end entity;
"""
    print(entity)

  def instantiate(self, unit_name, entity_name):
    # remove leading whitespace
    # the required leading whitespace is present in the string
    # and remove final character, which is a semi-colon
    self.instantiate_signals = self.instantiate_signals.lstrip()[:-1]

    entity = f"""

  {unit_name} : entity work.{entity_name}
    port(
      {self.instantiate_signals}
    );
"""
    return entity

