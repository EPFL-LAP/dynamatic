import argparse
import os
import re


def get_cli_arguments():
    parser = argparse.ArgumentParser(
        prog="Add opaque buffers",
        description="""Add opaque buffers following a component""",
    )

    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--ssa_name", type=str, required=True)
    parser.add_argument("--buffer_size", type=str, required=True)
    return parser.parse_args()


def get_bb(operation):
    match = re.search(r"handshake\.bb\s*=\s*(\d+)", operation)
    return int(match.group(1)) if match else 0


def get_ssa_name(operation):
    match = re.search(r"^    %(\d+)", operation)
    return "%" + match.group(1) if match else ""


def get_type(operation):
    bracket_match = re.search(r"<([^>]*)>$", operation)
    return bracket_match.group(1) if bracket_match else None


def get_opaque_buffer(ssa_result, ssa_input, line):

    opaque_buffer = """    %{ssa_result} = \
buffer {ssa_input} {{handshake.bb = {bb} : ui32, \
handshake.name = "buffer{ssa_result}", \
hw.parameters = {{NUM_SLOTS = 1 : ui32, \
TIMING = #handshake<timing {{D: 1, V: 1, R: 0}}>}}}} \
: <{type_operation}>
"""

    return opaque_buffer.format(
        ssa_result=ssa_result,
        ssa_input=ssa_input,
        bb=get_bb(line),
        type_operation=get_type(line),
    )


def look_for_ssa(lines, ssa_name):
    found_ssa_name = 0
    for line in lines:
        if ssa_name in line:
            found_ssa_name = 1
        if f'handshake.name = "{ssa_name}' in line:
            found_ssa_name = 2
    return found_ssa_name


def get_max_ssa(lines):
    result = 0
    for line in lines:
        match = re.search(r"^    %(\d+)", line)
        result = max(int(match.group(1)) if match else 0, result)
    return result


def process_file(file_name, ssa_name, buffer_size):

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: The file '{file_name}' does not exist.")

    with open(file_name, "r", encoding="utf-8") as file:
        lines = file.readlines()

    start_ssa_counter = get_max_ssa(lines)
    end_ssa_counter = start_ssa_counter + buffer_size
    new_lines = []

    found_ssa_name = look_for_ssa(lines, ssa_name)

    if found_ssa_name == 0:
        print(f"SSA name {ssa_name} cannot be found")
        exit(1)

    if found_ssa_name == 2:
        for line in lines:
            if f'handshake.name = "{ssa_name}' in line:
                ssa_name = get_ssa_name(line)

    for line in lines:
        new_lines.append(line)

        if f"{ssa_name} = " in line:
            for i in range(start_ssa_counter, end_ssa_counter):
                ssa_input = ssa_name if i == start_ssa_counter else f"%{i}"
                new_lines.append(get_opaque_buffer(i + 1, ssa_input, line))
        elif ssa_name in line:
            new_lines[-1] = line.replace(ssa_name, f"%{end_ssa_counter}")
            continue

    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(new_lines)


def main():
    args = get_cli_arguments()
    process_file(args.file_name, args.ssa_name, int(args.buffer_size))
    pass


if __name__ == "__main__":
    main()
