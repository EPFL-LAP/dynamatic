   
class DeclarativeUnit():
   """
   Class used to generate RTL by declaratively
   stating its input and output ports, and local signals

   Has one function "name",
   based on self.parent and self.unit_name

   Uses self.entity_port_items to print the entity port mapping,
   self.local_items to print the local signals,
   and self.body to print the body.

   self.top_level_comment is placed above the entity declaration
   """
   
   def name(self):
      return f"{self.parent}_{self.unit_name}"

class Entity():
  def __init__(self):
    self.entity_port_items = ""

  def __init__(self, declaration):
    self.entity_port_items = ""

    for item in declaration.entity_port_items:
        self.entity_port_items += item.get_entity_item()
      
    self.name = declaration.name()

    self.top_level_comment = declaration.top_level_comment
    


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

{self.top_level_comment}
entity {self.name}_unit is
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

      self.name = declaration.name()

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
architecture arch of {self.name}_unit is
  {self.local_items}
begin

  {self.body}
  
end architecture;
  """
      return architecture

class Instantiation():
    """
    Class to define RTL instantiations of other entities
    The python code matches the verbosity of RTL,
    so instantiations are described fully separately to the
    entity descriptions.
    """
    def __init__(self, unit_name, parent, port_items, comment=""):
      self.port_items = ""
      self.unit_name = unit_name
      self.parent = parent

      for port_item in port_items:
         self.port_items += port_item.get_inst_item()

      self.port_items = self.port_items.strip()[:-1]

      self.comment = comment
    
    def get(self):
       return f"""
  {self.comment}
  {self.unit_name}_unit : entity work.{self.parent}_{self.unit_name}_unit
    port map(
      {self.port_items}
    );

""".removeprefix("\n")