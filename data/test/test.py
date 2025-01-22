import os
import re
import subprocess
import argparse

# Path to nuXmv binary and command file
NUXMV_PATH = "/opt/nuXmv-2.0.0-Linux/bin/nuXmv"
CMD_FILE = "prove.cmd"

# Directory containing your modules
MODULES_DIR = "../smv"
MAIN_DIR = "./tmp"

# Ensure output directory exists
os.makedirs(MAIN_DIR, exist_ok=True)

def generate_main(module_name, module_path):
    """
    Generates a main module that instantiates the given module.
    """
    
    # Read the module file to search for parameters
    with open(module_path, "r") as file:
        content = file.read()
    
    # Regex to find the module declaration with parameters
    pattern = rf"MODULE\s+{module_name}\s*\((.*?)\)"
    match = re.search(pattern, content)
    if match:
        # Extract parameter names and provide dummy values
        params = match.group(1).replace(" ", "").split(",")  # Remove spaces and split by comma
        param_values = ", ".join(f"{param}" for param in params)
        decl_code = ""
        for param in params:
            decl_code += f"VAR {param} : boolean;\n"
        instance_code = f"instance : {module_name}({param_values});"
    else:
        # No parameters found
        decl_code = ""
        instance_code = f"instance : {module_name};"


    code = f"""
#include "../{module_path}"
MODULE main
{decl_code}
VAR
  {instance_code}
    """


    file_path = MAIN_DIR + "/main.smv"
    with open(file_path, "w") as file:
        file.write(code)
    return file_path

def run_nuxmv(main_path):
    """
    Runs nuXmv on the given module with the prove.cmd script.
    """
    try:
        result = subprocess.run(
            [NUXMV_PATH, "-source", CMD_FILE, main_path],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"nuXmv output for {main_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running nuXmv for {main_path}:\n{e.stderr}")

def find_module_file(module_name):
    """
    Searches for a file matching the module name in MODULES_DIR and its subdirectories.
    """
    for root, _, files in os.walk(MODULES_DIR):
        for file in files:
            if file == f"{module_name}.smv":
                return os.path.join(root, file)
    return None

def test_modules(module_name=None):
    """
    Tests a specific module by name or all modules if no name is provided.
    """
    if module_name:
        # Look for the specified module
        module_path = find_module_file(module_name)
        if not module_path:
            print(f"Error: Module '{module_name}' not found.")
            return
        print(f"Processing module: {module_name} in {module_path}")
        
        # Generate main module and run nuXmv
        main_path = generate_main(module_name, module_path)
        run_nuxmv(main_path)
    else:
        # Process all modules
        for root, _, files in os.walk(MODULES_DIR):
            for file in files:
                if file.endswith(".smv"):
                    module_name = os.path.splitext(file)[0]
                    module_path = os.path.join(root, file)
                    print(f"Processing module: {module_name} in {module_path}")
                    
                    # Generate main module and run nuXmv
                    main_path = generate_main(module_name, module_path)
                    run_nuxmv(main_path)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test nuXmv modules.")
    parser.add_argument(
        "module_name",
        nargs="?",
        default=None,
        help="Name of the module to test (without extension). If not provided, tests all modules."
    )
    args = parser.parse_args()

    # Tests the specified module or all modules
    test_modules(args.module_name)

if __name__ == "__main__":
    main()