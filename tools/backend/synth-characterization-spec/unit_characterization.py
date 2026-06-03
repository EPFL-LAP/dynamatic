from run_synthesis import run_synthesis, write_sdc_constraints
import os
import re
from itertools import product
from utils import parameters_ranges, VhdlInterfaceInfo, UnitCharacterization
from typing import List, Tuple


def extract_generics_ports(vhdl_code, entity_name):
    """
    Extract generics and ports from a VHDL entity block.
    Args:
        vhdl_code (str): VHDL code as a string.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of generics
            - List of ports
    """

    # Remove comments
    vhdl_code = re.sub(r'--.*', '', vhdl_code)

    # Match the entity block
    entity_match = re.search(
        r'entity\s+(\w+)\s+is(.*?)end\s+entity', vhdl_code, re.DOTALL | re.IGNORECASE)
    if not entity_match:
        raise ValueError("Could not find VHDL entity block.")

    # If there are multiple entities, we remove the one that does not match the entity_name
    is_right_entity = entity_match.group(1) in entity_name
    while not is_right_entity and entity_match:
        entity_to_remove = entity_match.group(1)
        # Remove the first entity block that does not match the entity_name
        first_entity = re.search(
            fr'entity\s+{entity_to_remove}\s+is(.*?)end\s+architecture', vhdl_code, re.DOTALL | re.IGNORECASE)
        assert first_entity, f"Could not find VHDL entity block for {entity_to_remove} despite being in the code."
        vhdl_code = vhdl_code.replace(first_entity.group(0), "")
        entity_match = re.search(
            r'entity\s+(\w+)\s+is(.*?)end\s+entity', vhdl_code, re.DOTALL | re.IGNORECASE)
        entity_extracted = entity_match.group(1)
        # Check if the entity name matches the one we are looking for # Selector is a special case since its handshake name is handshake.select
        is_right_entity = entity_extracted in entity_name if entity_extracted != "selector" else True

    assert entity_match, f"Entity {entity_name} not found in the VHDL code."

    entity_name = entity_match.group(1) if entity_match else "unknown_entity"
    entity_block = entity_match.group(2)

    # Extract generics and ports
    generics_match = re.search(
        r'generic\s*\((.*?)\)\s*;', entity_block, re.DOTALL | re.IGNORECASE)
    ports_match = re.search(
        r'port\s*\(((?:[^()]*|\([^()]*\))*)\)\s*;', entity_block, re.DOTALL | re.IGNORECASE)

    generics_raw = generics_match.group(1).strip() if generics_match else ''
    ports_raw = ports_match.group(1).strip() if ports_match else ''

    # Split on semicolon while keeping line breaks (in case of multiple declarations)
    def split_definitions(raw: str) -> List[str]:
        lines = re.split(r';\s*\n', raw)
        return [line.strip() for line in lines if line.strip()]

    generics = split_definitions(generics_raw)
    ports = split_definitions(ports_raw)

    return entity_name, VhdlInterfaceInfo(generics, ports)


def generate_wrapper_top(entity_name, vhdl_interface_info, param_names):
    """
    Generate the wrapper for the top file from the given top definition file.

    Args:
        entity_name (str): Name of the top entity.
        vhdl_interface_info (VhdlInterfaceInfo): VHDL interface information containing generics and ports.
        param_names (List[str]): List of parameter names to be used in the wrapper.

    Returns:
        str: The wrapper for the top file.
    """
    generics = vhdl_interface_info.get_list_generics()
    ports = vhdl_interface_info.get_list_ports()
    for _generic in generics:
        # Get the generic name before the colon
        generic = _generic.split(":")[0].strip()
        assert generic in param_names, f"Generic `{generic}` not found in parameter names."

    # Create ports for the top file
    tb_ports = []
    for _port in ports:
        port = _port
        for param in param_names:
            port = port.replace(f"{param}", f"{param}_const_value")
        tb_ports.append(port)
    # Create the wrapper for the top file
    wrapper_top = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;
entity tb is
port (
{';\n'.join(tb_ports)}
);
end entity;
architecture tb_arch of tb is
begin
dut: entity work.{entity_name}
generic map (
"""

    for param in param_names:
        if param != "PREDICATE":
            wrapper_top += f"{param} => {param}_const_value,\n"
    wrapper_top = wrapper_top.rstrip(",\n") + "\n" \
        ")\n" \
        "port map (\n"
    for port in ports:
        # Get the port name before the colon
        port_name = port.split(":")[0].strip()
        wrapper_top += f"{port_name} => {port_name},\n"
    wrapper_top = wrapper_top.rstrip(",\n") + "\n" \
        ");\n" \
        "end architecture;\n"

    return wrapper_top, entity_name


def generate_wrapper_top_no_generics(entity_name, vhdl_interface_info):
    """
    Wrapper used when the unit's VHDL has no generics (modern self-contained generator output).
    The tb just wires every dut port to a top-level port of the same name.
    """
    ports = vhdl_interface_info.get_list_ports()
    tb_ports = list(ports)

    wrapper_top = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;
entity tb is
port (
{';\n'.join(tb_ports)}
);
end entity;
architecture tb_arch of tb is
begin
dut: entity work.{entity_name}
port map (
"""
    for port in ports:
        port_name = port.split(":")[0].strip()
        wrapper_top += f"{port_name} => {port_name},\n"
    wrapper_top = wrapper_top.rstrip(",\n") + "\n);\nend architecture;\n"
    return wrapper_top


def run_generator_for_combo(generator, dynamatic_dir, hdl_out_dir, module_name, params_dict):
    """
    Substitute $DYNAMATIC / $OUTPUT_DIR / $MODULE_NAME / each $<PARAM_NAME> into the
    generator command template and invoke it. Returns the path of the produced .vhd.
    """
    cmd = generator
    cmd = cmd.replace("$DYNAMATIC", dynamatic_dir)
    cmd = cmd.replace("$OUTPUT_DIR", hdl_out_dir)
    cmd = cmd.replace("$MODULE_NAME", module_name)
    for param_name, param_value in params_dict.items():
        cmd = cmd.replace(f"${param_name}", str(param_value))
    print(f"Running generator command: {cmd}")
    rc = os.system(cmd)
    rtl_file = f"{hdl_out_dir}/{module_name}.vhd"
    assert rc == 0, f"Generator command failed for {module_name}: {cmd}"
    assert os.path.exists(rtl_file), f"Generator did not produce {rtl_file}"
    return rtl_file


def run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir, clock_period, generator=None, dynamatic_dir=None):
    """
    Run characterization for a single unit using the specified synthesis tool.

    When `generator` is provided, the generator is re-invoked per parameter
    combination with that combo's values substituted into the command, producing
    a fresh fully-specialised .vhd for each combo. Otherwise (generic-only
    units), the existing top_def_file is used and the legacy VHDL-generic
    sweeping path is taken.
    """
    support_hdl_files = [f"{hdl_out_dir}/{file}" for file in os.listdir(
        hdl_out_dir) if os.path.isfile(os.path.join(hdl_out_dir, file))]

    params_charact = {}
    for param in list_params:
        param_name = param["name"]
        # Only sweep params that the generator command actually references;
        # avoids a redundant cartesian product when the JSON declares a param
        # the generator does not use (e.g. DATA_TYPE alongside BITWIDTH).
        if generator is not None and f"${param_name}" not in generator:
            continue
        assert param_name in parameters_ranges, f"Parameter {param_name} not found in parameters_ranges."
        params_charact[param_name] = parameters_ranges[param_name]
    param_combinations = list(
        product(*params_charact.values())) if params_charact else [()]
    param_names = list(params_charact.keys())

    sdc_file = f"{tcl_dir}/period.sdc"
    write_sdc_constraints(sdc_file, clock_period)

    list_tcls = []
    unit_characterization_list = []

    if generator is not None:
        for id, combination in enumerate(param_combinations):
            params_dict = dict(zip(param_names, combination))
            module_name = f"{unit_name.split('.')[-1]}_{id}"
            unit_vhd = run_generator_for_combo(
                generator, dynamatic_dir, hdl_out_dir, module_name, params_dict)
            with open(unit_vhd, 'r') as f:
                vhdl_code = f.read()
            top_entity_name, vhdl_interface_info = extract_generics_ports(
                vhdl_code, module_name)
            tb_text = generate_wrapper_top_no_generics(
                top_entity_name, vhdl_interface_info)
            tb_file = f"{hdl_out_dir}/{module_name}_tb.vhd"
            with open(tb_file, 'w') as f:
                f.write(tb_text)
            hdl_files = [tb_file, unit_vhd] + support_hdl_files
            unit_char_obj = UnitCharacterization(
                unit_name, top_entity_name, params_dict, hdl_files, vhdl_interface_info, id)
            list_tcls.append(unit_char_obj.generate_tcl(
                tcl_dir, rpt_dir, sdc_file))
            unit_characterization_list.append(unit_char_obj)
    else:
        print(f"Extracting generics and ports from {top_def_file}")
        with open(top_def_file, 'r') as f:
            vhdl_code = f.read()
        top_entity_name, vhdl_interface_info = extract_generics_ports(
            vhdl_code, unit_name)
        wrapper_top, top_entity_name = generate_wrapper_top(
            top_entity_name, vhdl_interface_info, param_names)
        for id, combination in enumerate(param_combinations):
            top_file = f"{hdl_out_dir}/{top_entity_name}_top_{id}.vhd"
            wrapper_top_combined = wrapper_top
            for param_name, param_value in zip(param_names, combination):
                wrapper_top_combined = wrapper_top_combined.replace(
                    f"{param_name}_const_value", str(param_value))
            with open(top_file, 'w') as f:
                f.write(wrapper_top_combined)
            unit_char_obj = UnitCharacterization(unit_name, top_entity_name, dict(zip(
                param_names, combination)), [top_file] + support_hdl_files, vhdl_interface_info, id)
            list_tcls.append(unit_char_obj.generate_tcl(
                tcl_dir, rpt_dir, sdc_file))
            unit_characterization_list.append(unit_char_obj)

    log_file = f"{log_dir}/synth_{unit_name}_log.txt"
    run_synthesis(list_tcls, synth_tool, log_file)

    return unit_characterization_list
