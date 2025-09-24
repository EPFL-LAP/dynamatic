import json
import argparse

input_path= "/beta/rouzbeh/fpga-main/dynamatic//data/rtl-config-vhdl copy.json"
output_path= "/beta/rouzbeh/fpga-main/dynamatic//data/rtl-config-vhdl.json"
# Function to replace placeholders in the generator string
def update_json_file(load_value, store_value):
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Update the generator string in a JSON file.")
    parser.add_argument('load', type=int, help="The value to replace $LOAD")
    parser.add_argument('store', type=int, help="The value to replace $STORE")
    
    args = parser.parse_args()
    
    # Call the function to update the file
    update_json_file(args.load, args.store)

if __name__ == "__main__":
    main()