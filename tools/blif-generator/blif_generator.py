import os
import subprocess
import argparse
import sys
import json
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any
from itertools import product
import re

# Path configuration
DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = DYNAMATIC_ROOT / "data"
JSON_FILE_PATH = os.path.join(DATA_FOLDER, 'rtl-config-verilog.json')
BLIF_FILES_PATH = os.path.join(DATA_FOLDER, 'blif')
BLACKBOX_COMPONENTS = ['addi', 'cmpi', 'subi', 'muli', 'divsi', 'divui']

def create_yosys_script(component: str, params: str, output_dir: str, output_name: str, verilog_files: List[str]) -> str:
    """Create a Yosys script with the given parameters."""
    verilog_reads = '\n        '.join([f'read_verilog -defer {path}' for path in verilog_files])

    content = f"""#!/bin/bash
yosys -p "{verilog_reads}
        {params}
        hierarchy -top {component};
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif {output_dir}/{output_name}" > /dev/null
"""
    return content

def create_abc_script(input_file: str, output_dir: str, output_name: str) -> str:
    """Create an ABC script with optimization commands."""
    optimization_steps = ['strash'] + ['rewrite', 'b', 'refactor', 'b'] * 6
    commands = ';\n        '.join(optimization_steps)

    content = f"""#!/bin/bash
abc -c "read_blif {input_file};
        {commands};
        write_blif {output_dir}/{output_name}"
"""
    return content


def get_range_for_param(param_type: str) -> Tuple[int, int]:
    """
    Get the range for a given parameter type.
    Range of iteration for each parameter.
    """
    ranges = {
        'NUM_SLOTS': (1, 10),
        'SIZE': (1, 15),
        'INDEX_TYPE': (1, 4),
        'SELECT_TYPE': (1, 4),
        'DEFAULT': (1, 33)
    }
    start, end = ranges.get(param_type, ranges['DEFAULT'])
    return start, end

def load_json_config(config_path: str) -> List[Dict[str, Any]]:
    """Load the JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_verilog_dependencies(module_config: Dict[str, Any], all_modules: List[Dict[str, Any]]) -> List[str]:
    """Get list of Verilog dependency files for a module by looking up in JSON config, including recursive dependencies."""
    verilog_paths = []
    visited_dependencies = set()  # To avoid circular dependencies
    
    # Create a lookup dict for module-name to config mapping
    module_name_lookup = {}
    for config in all_modules:
        if 'module-name' in config:
            module_name_lookup[config['module-name']] = config
        # Also add entries without module-name using their base name
        elif 'name' in config:
            base_name = config['name'].replace('handshake.', '')
            module_name_lookup[base_name] = config
        # Also add entries that have only 'generic' field (support modules)
        elif 'generic' in config:
            # Extract filename without extension and path for support modules
            generic_path = config['generic']
            filename = os.path.basename(generic_path).replace('.v', '')
            module_name_lookup[filename] = config
    
    def collect_dependencies_recursive(config: Dict[str, Any]) -> None:
        """Recursively collect all dependencies for a given config."""
        # Add the main generic file if it exists
        if 'generic' in config:
            generic_path = config['generic'].replace('$DYNAMATIC', str(DYNAMATIC_ROOT))
            if os.path.exists(generic_path) and generic_path not in verilog_paths:
                verilog_paths.append(generic_path)
        
        # Process dependencies
        dependencies = config.get('dependencies', [])
        for dep in dependencies:
            if dep in visited_dependencies:
                continue  # Skip already processed dependencies to avoid circular references
            
            visited_dependencies.add(dep)
            
            if dep in module_name_lookup:
                dep_config = module_name_lookup[dep]
                # Recursively collect dependencies of this dependency
                collect_dependencies_recursive(dep_config)
            else:
                print(f"Warning: Dependency '{dep}' not found in JSON configuration")
    
    # Start recursive collection from the main module
    collect_dependencies_recursive(module_config)
    
    return verilog_paths

def get_parameter_ranges(module_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Get parameter ranges for a module based on its configuration."""
    parameters = module_config.get('parameters', [])
    param_ranges = {}
    
    for param in parameters:
        param_name = param['name']
        param_type = param['type']

        if param.get('generic', True): 
            if param_type == 'string':
                if 'eq' in param:
                    param_ranges[param_name] = [param['eq']]
                else:
                    # Do not add to list
                    pass
            elif param_type in ['unsigned', 'dataflow']:
                # Check for fixed values first
                if 'eq' in param:
                    param_ranges[param_name] = [param['eq']]
                elif 'data-eq' in param:
                    param_ranges[param_name] = [param['data-eq']]
                else:
                    # Get range from get_range_for_param function
                    start, end = get_range_for_param(param_name)
                    
                    # Apply lower bound if specified
                    if 'lb' in param:
                        start = max(start, param['lb'])
                    if 'data-lb' in param:
                        start = max(start, param['data-lb'])
                    
                    param_ranges[param_name] = list(range(start, end))
            elif param_type == 'timing':
                # Check for fixed timing values
                if 'eq' in param:
                    param_ranges[param_name] = [param['eq']]
                elif 'data-lat-eq' in param:
                    param_ranges[param_name] = [param['data-lat-eq']]
                elif 'valid-lat-eq' in param:
                    param_ranges[param_name] = [param['valid-lat-eq']]
                elif 'ready-lat-eq' in param:
                    param_ranges[param_name] = [param['ready-lat-eq']]
                # Skip timing parameters that don't have fixed values
                continue
    
    return param_ranges

def execute_generator(generator_cmd: str, module_name: str, output_dir: str, **kwargs) -> str:
    """Execute a generator command and return the generated Verilog file path."""
    # Replace placeholders in generator command
    cmd = generator_cmd.replace('$DYNAMATIC', str(DYNAMATIC_ROOT))
    cmd = cmd.replace('$OUTPUT_DIR', output_dir)
    cmd = cmd.replace('$MODULE_NAME', module_name)
    
    if (module_name == "constant"):
        cmd = cmd.replace('$VALUE', '1')
    
    if (module_name == "cmpi"):
        cmd = cmd.replace('$PREDICATE', 'ult')

    # Replace any additional parameters
    for key, value in kwargs.items():
        cmd = cmd.replace(f'${key}', str(value))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute generator
    try:
        subprocess.run(cmd, shell=True, check=True)
        generated_file = os.path.join(output_dir, f'{module_name}.v')
        return generated_file if os.path.exists(generated_file) else None
    except subprocess.CalledProcessError as e:
        print(f"Generator failed for {module_name}: {e}")
        return None

def process_blackbox_components(file_path: str, component: str, data_type: int = None) -> None:
    """Filter BLIF file based on component-specific rules."""
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        should_keep = True
        should_skip_next = False

        # Remove .names lines for these modules, as they are blackboxes
        if (component in ['addi', 'cmpi', 'subi'] and 
            data_type is not None and data_type > 4):
            if '.names' in line and 'ready' not in line and 'valid' not in line:
                should_keep = False
                should_skip_next = True  # Skip the line after .names

        # Rules for muli, divsi, divui (all DATA_TYPEs)
        elif component in ['muli', 'divsi', 'divui']:
            if '.names' in line:
                should_keep = False
                should_skip_next = True  # Skip the line after .names
            elif '.latch' in line:
                should_keep = False
                # Don't skip next line for .latch

        if should_keep:
            filtered_lines.append(line)
        elif should_skip_next and i + 1 < len(lines):
            skip_next = True

    # Write filtered content back to file
    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)

    if component == 'muli':
        copy_and_rename_blif('muli', 'divsi', data_type)
        copy_and_rename_blif('muli', 'divui', data_type)

def copy_and_rename_blif(source_component: str, target_component: str, data_type: int) -> None:
    """Copy BLIF file from source component to target component with name updates."""
    source_dir = os.path.join(BLIF_FILES_PATH, source_component, str(data_type))
    target_dir = os.path.join(BLIF_FILES_PATH, target_component, str(data_type))

    source_file = os.path.join(source_dir, f'{source_component}.blif')
    target_file = os.path.join(target_dir, f'{target_component}.blif')

    if not os.path.exists(source_file):
        print(f"Warning: Source file {source_file} does not exist")
        return

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Read source file and replace component names
    with open(source_file, 'r') as f:
        content = f.read()

    # Replace all occurrences of source component name with target component name
    updated_content = content.replace(source_component, target_component)

    # Write to target file
    with open(target_file, 'w') as f:
        f.write(updated_content)

def create_and_run_scripts(component: str, folder_path: str, param_command: str = '',
                          name_suffix: str = '', verilog_files: List[str] = None, data_type: int = None) -> None:
    """Create and execute Yosys and ABC scripts for a component."""
    if not verilog_files:
        verilog_files = []
    
    os.makedirs(folder_path, exist_ok=True)

    # Yosys script
    yosys_output = f'{component}{name_suffix}_yosys.blif'
    yosys_script_path = os.path.join(folder_path, 'run_yosys.sh')
    yosys_content = create_yosys_script(component, param_command, folder_path, yosys_output, verilog_files)

    with open(yosys_script_path, 'w') as f:
        f.write(yosys_content)
    os.chmod(yosys_script_path, 0o755)
    subprocess.run(['bash', yosys_script_path])

    # ABC script
    abc_output = f'{component}.blif'
    abc_input = os.path.join(folder_path, yosys_output)
    abc_script_path = os.path.join(folder_path, 'run_abc.sh')
    abc_content = create_abc_script(abc_input, folder_path, abc_output)

    with open(abc_script_path, 'w') as f:
        f.write(abc_content)
    os.chmod(abc_script_path, 0o755)
    subprocess.run(['bash', abc_script_path])

    abc_output_path = os.path.join(folder_path, abc_output)
    
    # Process blackbox components after ABC
    if component in BLACKBOX_COMPONENTS:
        process_blackbox_components(abc_output_path, component, data_type)

def process_module(module_config: Dict[str, Any], all_modules: List[Dict[str, Any]]) -> None:
    """Process a single module based on its JSON configuration."""
    module_name = module_config['name']
    
    # Get module name override if specified
    if 'module-name' in module_config:
        component = module_config['module-name']
    else:
        # Extract component name (remove handshake. prefix if present)
        component = module_name.replace('handshake.', '')
    
    print(f"Processing module: {module_name} (component: {component})")
    
    # Get parameter ranges and generic parameters
    param_ranges = get_parameter_ranges(module_config)

    parameters = module_config.get('parameters', [])
    generic_params = []
    for param in parameters:
        param_name = param['name']
        # Get list of parameter names that should be passed to Yosys
        if param.get('generic', True):  # Default to True if not specified
            generic_params.append(param_name)

    # Get Verilog dependencies
    verilog_files = get_verilog_dependencies(module_config, all_modules)

    base_folder_path = os.path.join(BLIF_FILES_PATH, component)
    
    # If no parameters, process once
    if not param_ranges:
        create_and_run_scripts(component, base_folder_path, '', '', verilog_files)
        return
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    for combo in product(*param_values):
        # Handle generator case
        if 'generator' in module_config:
            # Create folder for this parameter combination
            folder_path = os.path.join(base_folder_path, *map(str, combo))
            
            # Create parameter dict for generator
            param_dict = dict(zip(param_names, combo))
            
            # Execute generator
            generated_file = execute_generator(
                module_config['generator'], 
                component, 
                folder_path,
                **param_dict
            )
            
            if generated_file:
                verilog_files_with_generated = verilog_files + [generated_file]
            else:
                print(f"Failed to generate file for {component} with params {combo}")
                continue
        else:
            folder_path = os.path.join(base_folder_path, *map(str, combo))
            verilog_files_with_generated = verilog_files
        
        # Create parameter command for Yosys - only include generic parameters
        generic_param_commands = []
        for i, param_name in enumerate(param_names):
            if param_name in generic_params:
                generic_param_commands.append(f'-set {param_name} {combo[i]}')
        
        if generic_param_commands:
            param_command = f"chparam {' '.join(generic_param_commands)} {component}"
        else:
            param_command = ''
        
        name_suffix = '_' + '_'.join(str(val) for val in combo)
        
        # Get data_type for blackbox processing
        data_type = None
        if 'DATA_TYPE' in param_names:
            data_type_index = param_names.index('DATA_TYPE')
            data_type = combo[data_type_index]
        
        create_and_run_scripts(component, folder_path, param_command, name_suffix, verilog_files_with_generated, data_type)

def main():
    parser = argparse.ArgumentParser(description='Generate AIGs for modules using JSON configuration')
    parser.add_argument('module', nargs='?', help='Specific module name to generate (optional)')

    args = parser.parse_args()

    # Load JSON configuration
    try:
        modules_config = load_json_config(JSON_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Configuration file '{JSON_FILE_PATH}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

    # Create a lookup dict for modules
    modules_dict = {config['name']: config for config in modules_config if 'name' in config}

    if args.module:
        if args.module not in modules_dict:
            print(f"Error: Module '{args.module}' not found!")
            sys.exit(1)
        print(f"Processing module: {args.module}")
        process_module(modules_dict[args.module], modules_config)
    else:
        print("Processing all modules...")
        for module_config in modules_config:
            if 'name' in module_config:  # Only process modules with names
                if not re.search(r'[^/]*f$', module_config['name']): # Skip floating point units
                    process_module(module_config, modules_config)

    print("Processed modules")

if __name__ == "__main__":
    main()
