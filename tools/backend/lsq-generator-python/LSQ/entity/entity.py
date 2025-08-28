


# def makeEntitySignal(base_name, signal):
   
    


# def makeInstantiationSignal(base_name, signal):
#     name = f"{base_name}"

#     full_declaration = f"{name},"
#     if signal.signal_size.bitwidth == 0:
#         full_declaration = f"-- {full_declaration}"
#     return f"""
#       {full_declaration}
# """.removeprefix("\n")
    

class Entity():
  def __init__(self):
    self.entity_port_items = ""

  def __init__(self, declaration):
    self.entity_port_items = ""

    for item in declaration.entity_port_items:
        self.entity_port_items += item.get_entity_item()


  def get(self, name, entity_type):
    # remove leading whitespace
    # the required leading whitespace is present in the string
    # and remove final character, which is a semi-colon
    self.entity_port_items = self.entity_port_items.strip()
    self.entity_port_items = self.entity_port_items[:-1]

    entity = f"""
-- {entity_type}
entity {name} is
  port(
    {self.entity_port_items}
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

class Architecture():
    def __init__(self, declaration):
      self.local_items = ""

      for item in declaration.local_items:
          self.local_items += item.get_local_item()

    def get(self, name, entity_type):
      # remove leading whitespace
      # the required leading whitespace is present in the string
      # and remove final character, which is a semi-colon
      self.local_items = self.local_items.strip()
      self.local_items = self.local_items[:-1]

      architecture = f"""
  -- {entity_type}
architecture arch of {name} is
  {self.local_items}
begin architecture
end architecture;
  """
      print(architecture)