import json
import argparse

input_path= "/beta/rouzbeh/clean/dynamatic//data/rtl-config-vhdl copy.json"
output_path= "/beta/rouzbeh/clean/dynamatic//data/rtl-config-vhdl.json"


def change_lsq_size(load_value, store_value):
    # Open the original JSON file
    with open(input_path, 'r') as file:
        data = json.load(file)
    
    new_data = []
    for d in data:
      if d.get("name") == "handshake.lsq":
        # Get the generator string from the JSON data
        generator_string = d.get("generator", "")
        
        # Replace the placeholders with the input values
        updated_generator = generator_string.replace("$LOAD", str(load_value)).replace("$STORE", str(store_value))
        
        # Update the JSON data with the new generator string
        d["generator"] = updated_generator
        

      new_data.append(d)
    
    # Write the updated data back to a new JSON file
    with open(output_path, 'w') as file:
        json.dump(new_data, file, indent=4)