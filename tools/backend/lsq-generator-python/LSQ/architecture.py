from LSQ.entity import signalSizeToTypeDeclaration, SignalSize

def makeArchitectureSignal(base_name, signal):
  name = base_name.ljust("20")
  type_declaration = signalSizeToTypeDeclaration(signal.signal_size)
  return f"""
  signal {name} : {type_declaration};
""".removeprefix("\n")

class Architecture():
  def __init__(self, declaration):
    self.signals = ""
    for signal in declaration.local_signals:
      self._addSignal(signal)

  def _addSignal(self, signal):
    if signal.comment is not None:
      self.signals += signal.comment
        
    if signal.signal_size.number == 1:
        newSignal = makeArchitectureSignal(
          signal.rtl_name,
          signal
        )
        self.signals += newSignal
    else:
        for i in range(signal.signal_size.number):
          newSignal = makeArchitectureSignal(
              f"{signal.rtl_name}_{i}",
              signal
            )
        self.signals += newSignal

    
  def get(self, name):
    # remove leading whitespace
    # the required leading whitespace is present in the string
    # and remove final character, which is a semi-colon
    self.signals = self.signals.lstrip()[:-1]

    architecture = f"""
architecture arch of {name} is
    {self.signals}
begin
end architecture;
"""
    print(architecture)