import os
import subprocess
import argparse
import sys
import re
import glob
from pathlib import Path
from typing import Tuple
from itertools import product

# Path configuration
DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = DYNAMATIC_ROOT / "data"
BLIF_FILES_PATH = os.path.join(DATA_FOLDER, 'blif')

VERILOG_PATHS = (
    [f for f in glob.glob(f'{DATA_FOLDER}/verilog/arith/*.v') if not re.search(r'[^/]*f\.v$', f)] # Don't read floating point modules
    + glob.glob(f'{DATA_FOLDER}/verilog/handshake/*.v')
    + glob.glob(f'{DATA_FOLDER}/verilog/handshake/dataless/*.v')
    + glob.glob(f'{DATA_FOLDER}/verilog/support/*.v')
    + glob.glob(f'{DATA_FOLDER}/verilog/support/dataless/*.v')
)

# Remove files that causes issues with Yosys
EXCLUDE_FILES = {
    f'{DATA_FOLDER}/verilog/arith/cmpi.v',
    f'{DATA_FOLDER}/verilog/handshake/dataless/ndwire.v'
}

VERILOG_PATHS = [f for f in VERILOG_PATHS if f not in EXCLUDE_FILES]

BLACKBOX_MODULES = ['addi', 'cmpi', 'subi', 'divsi', 'divui', 'muli']

ARITH_MODULES = {
    'addi': ['DATA_TYPE'], 'andi': ['DATA_TYPE'],
    'cmpi': ['DATA_TYPE'], 'divsi': ['DATA_TYPE'],
    'divui': ['DATA_TYPE'], 'extsi': ['INPUT_TYPE', 'OUTPUT_TYPE'],
    'extui': ['INPUT_TYPE', 'OUTPUT_TYPE'], 
    'muli': ['DATA_TYPE'], 'ori': ['DATA_TYPE'],
    'select': [], 'shli': ['DATA_TYPE'], 'shrsi': ['DATA_TYPE'],
    'shrui': ['DATA_TYPE'], 'subi': ['DATA_TYPE'],
    'trunci': ['INPUT_TYPE', 'OUTPUT_TYPE'], 'xori': ['DATA_TYPE']
}

HANDSHAKE_MODULES = {
    'br': ['DATA_TYPE'], 'cond_br': ['DATA_TYPE'], 'constant': ['DATA_WIDTH'],
    'control_merge': ['SIZE', 'DATA_TYPE', 'INDEX_TYPE'], 'control_merge_dataless': ['SIZE', 'INDEX_TYPE'],
    'fork_type': ['SIZE', 'DATA_TYPE'], 'fork_dataless': ['SIZE'],
    'lazy_fork': ['SIZE', 'DATA_TYPE'], 'lazy_fork_dataless': ['SIZE'],
    'load': ['DATA_TYPE', 'ADDR_TYPE'],
    'merge': ['SIZE', 'DATA_TYPE'], 'merge_dataless': ['SIZE'],
    'mux': ['SIZE', 'DATA_TYPE', 'SELECT_TYPE'], 
    'oehb': ['DATA_TYPE'], 'oehb_dataless': [], 
    'one_slot_break_dvr': ['DATA_TYPE'], 'one_slot_break_dvr_dataless': [],
    'sink': ['DATA_TYPE'], 'sink_dataless': [], 'source': [],
    'br_dataless': [], 'cond_br_dataless': [], 'store': ['DATA_TYPE', 'ADDR_TYPE'],
    'tehb': ['DATA_TYPE'], 'tehb_dataless': []
}

def get_range_for_param(param_type: str) -> Tuple[int, int]:
    """
    Get the range for a given parameter type.
    Range of iteration for each parameter.
    """
    ranges = {
        'NUM_SLOTS': (1, 10),
        'SIZE': (2, 10),
        'INDEX_TYPE': (1, 4),
        'SELECT_TYPE': (1, 4),
        'DEFAULT': (1, 33)
    }
    start, end = ranges.get(param_type, ranges['DEFAULT'])
    return start, end

def create_yosys_script(component: str, params: str, output_dir: str, output_name: str) -> str:
    """Create a Yosys script with the given parameters."""
    verilog_reads = '\n        '.join([f'read_verilog -defer {path}' for path in VERILOG_PATHS])
    
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
                          name_suffix: str = '', data_type: int = None) -> None:
    """Create and execute Yosys and ABC scripts for a component."""
    # Skip divsi and divui - they will be copied from muli
    if component in ['divsi', 'divui']:
        return
    
    os.makedirs(folder_path, exist_ok=True)
    
    # Yosys script
    yosys_output = f'{component}{name_suffix}_yosys.blif'
    yosys_script_path = os.path.join(folder_path, 'run_yosys.sh')
    yosys_content = create_yosys_script(component, param_command, folder_path, yosys_output)
    
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

    if (component in BLACKBOX_MODULES):
        process_blackbox_components(abc_output_path, component, data_type)

def process_modules(target_component):
    """Process components with two parameters."""    
    # components_to_process = {target: DUAL_PARAM_MODULES[target] for target in target_component if target in DUAL_PARAM_MODULES}

    for component, params in target_component.items():
        param_command = ''
        base_folder_path = os.path.join(BLIF_FILES_PATH, component)

        if len(params) == 0:
            create_and_run_scripts(component, folder_path)
            continue

        ranges = [get_range_for_param(param) for param in params]

        for combo in product(*[range(start, end) for start, end in ranges]):
            param_command = f"chparam {' '.join(f'-set {params[i]} {val}' for i, val in enumerate(combo))} {component}"
            folder_path = os.path.join(base_folder_path, *map(str, combo))
            name_suffix = '_' + '_'.join(str(val) for val in combo)
            data_type = combo[-1] if 'DATA_TYPE' in params else None
            create_and_run_scripts(component, folder_path, param_command, name_suffix, data_type)


def main():
    parser = argparse.ArgumentParser(description='Generate AIGs for modules')
    parser.add_argument('module', nargs='?', help='Specific module name to generate (optional)')
    parser.add_argument('--list', action='store_true', help='List all available components')
    
    args = parser.parse_args()
    
    all_modules = ARITH_MODULES | HANDSHAKE_MODULES

    if args.list:
        print("Available components:")
        for mod in sorted(all_modules):
            print(f"  {mod}")
        return
    
    if args.module:
        if args.module not in all_modules.keys():
            print(f"Error: Module '{args.module}' not found!")
            sys.exit(1)
        print(f"Processing modules: {args.module}")
        process_modules({args.module: all_modules[args.module]})
    else:
        print("Processing all modules...")
        process_modules(all_modules)

    print("Processed modules")

if __name__ == "__main__":
    main()