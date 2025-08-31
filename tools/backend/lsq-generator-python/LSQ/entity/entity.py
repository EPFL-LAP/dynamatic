


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
      
    self.name = declaration.name
    self.prefix = declaration.prefix


  def get(self):
    # remove leading whitespace
    # the required leading whitespace is present in the string
    # and remove final character, which is a semi-colon
    self.entity_port_items = self.entity_port_items.strip()
    self.entity_port_items = self.entity_port_items[:-1]

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.types.all;

entity {self.prefix}_{self.name} is
  port(
    {self.entity_port_items}
  );
end entity;
"""
    return entity

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
      self.name = declaration.name

      self.prefix = declaration.prefix

      self.local_items = ""

      self.body = ""
      for body_item in declaration.body:
        self.body += body_item.get()

      self.body = self.body.strip()

      for item in declaration.local_items:
          self.local_items += item.get_local_item()

      # remove leading whitespace
      # the required leading whitespace is present in the string
      self.local_items = self.local_items.lstrip()

    def get(self):
      architecture = f"""
architecture arch of {self.prefix}_{self.name} is
  {self.local_items}
begin

  {self.body}
  
end architecture;
  """
      return architecture

class Instantiation():
    def __init__(self, name, prefix, port_items):
      self.port_items = ""
      self.name = name
      self.prefix = prefix

      for port_item in port_items:
         self.port_items += port_item.get_inst_item()

      self.port_items = self.port_items.strip()[:-1]
    
    def get(self):
       return f"""
  {self.name} : work.{self.prefix}_{self.name}
    port map(
      {self.port_items}
    );

""".removeprefix("\n")
