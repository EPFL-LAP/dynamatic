"""
wlf2csv.py 

Converts the waveform output of Modelsim (*.wlf file) into a simpler to parse CSV file
containing the subset of events that we care about for the dataflow visualizer. The
input DOT file that was used to generate the VHDL design whose simulation resulted in
the WLF file is used to map wire events back to the dataflow components they correspond
to. 
"""

import os
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

DATAIN_WIRE: str = "_dataInArray_"
VALID_WIRE: str = "_validArray_"
READY_WIRE: str = "_readyArray_"
COLOR_ATTR: str = "color"


class WireType(Enum):
    VALID = 0
    READY = 1
    DATA = 2


class WireState(Enum):
    LOGIC_0 = 0
    LOGIC_1 = 1
    UNDEFINED = 2

    @staticmethod
    def from_log(token: str) -> "WireState":
        val: str = token[2:-3]
        if val == "1":
            return WireState.LOGIC_1
        elif val == "0":
            return WireState.LOGIC_0
        return WireState.UNDEFINED


@dataclass(frozen=True)
class Port:
    component: str
    port_id: int
    is_input: bool


@dataclass(frozen=True)
class Channel:
    src: Port
    dst: Port
    datawidth: int


@dataclass
class DOTNode:
    name: str
    attributes: dict[str, str]

    def __str__(self) -> str:
        attr: list[str] = [f'{name}="{val}"' for (name, val) in self.attributes.items()]
        return f"\"{self.name}\" [{' '.join(attr)}]\n"


@dataclass
class DOTEdge:
    channel: Channel
    attributes: dict[str, str]
    src_dot_name: str
    dst_dot_name: str

    def __str__(self) -> str:
        attr: list[str] = [f'{name}="{val}"' for (name, val) in self.attributes.items()]
        return (
            f"\"{self.src_dot_name}\" -> \"{self.dst_dot_name}\" [{' '.join(attr)}]\n"
        )


class DOTGraph:
    name: str
    nodes: dict[str, DOTNode]
    edges: list[DOTEdge]

    port_to_edge: dict[Port, DOTEdge]

    def __init__(self, filepath: str) -> None:
        self.name = Path(filepath).stem
        self.nodes = {}
        self.edges = []
        self.port_to_edge = {}


class ChannelState:
    valid: WireState
    ready: WireState
    data: list[WireState]

    def __init__(self, datawidth: int) -> None:
        self.valid = WireState.UNDEFINED
        self.ready = WireState.UNDEFINED
        self.data = [WireState.UNDEFINED for _ in range(datawidth)]

    def data_to_dec(self) -> str:
        if len(self.data) == 0:
            return ""
        power: int = 1 << (len(self.data) - 1)
        value: int = 0
        for bit in self.data:
            if bit == WireState.UNDEFINED:
                return "undefined"
            if bit == WireState.LOGIC_1:
                value += power
            power >>= 1
        return str(value)


@dataclass
class WireReference:
    channel: Channel
    wire_type: WireType
    data_idx: int | None = None

    @staticmethod
    def from_signal(graph: DOTGraph, signal_name: str) -> "WireReference | None":
        # Take off the duv/ prefix
        name: str = signal_name[4:]

        to_channel = lambda port: graph.port_to_edge[port].channel

        try:
            # Derive port ID
            port_idx_str: str = name[name.rfind("_") + 1 :]
            if port_idx_str.endswith(")"):
                # This is a data signal
                port_id = int(port_idx_str[: port_idx_str.find("(")])
                name_idx = name.find(DATAIN_WIRE)
                assert name_idx != -1
                return WireReference(
                    to_channel(Port(name[:name_idx], port_id, True)),
                    WireType.DATA,
                    int(
                        port_idx_str[
                            port_idx_str.find("(") + 1 : port_idx_str.find(")")
                        ]
                    ),
                )

            port_id: int = int(port_idx_str)
            # Check if it is a valid signal
            if (name_idx := name.find(VALID_WIRE)) != -1:
                return WireReference(
                    to_channel(Port(name[:name_idx], port_id, False)), WireType.VALID
                )

            # Ready signal, by default
            name_idx = name.find(READY_WIRE)
            assert name_idx != -1
            return WireReference(
                to_channel(Port(name[:name_idx], port_id, True)), WireType.READY
            )
        except:
            return None


class DOTParsingError(Exception):
    def __init__(self, msg: str, line: str) -> None:
        super().__init__(f"{msg}\n\tIn: {line}")


def is_subgraph_decl(line: str) -> bool:
    tokens = line.strip().split(" ")
    return tokens[0] == "subgraph"


def find_outside_quotes(
    txt: str, find: str, start: int | None = None, end: int | None = None
) -> int:
    start_idx: int = 0 if start is None else start
    end_idx: int = len(txt) if end is None else end
    in_quotes: bool = False
    for i, char in enumerate(txt[start_idx:end_idx]):
        if not in_quotes and txt[start_idx + i].startswith(find):
            return start_idx + i
        if char == '"':
            in_quotes = not in_quotes
    return -1


def has_attribute_list(line: str) -> tuple[int, int] | None:
    open_bracket = find_outside_quotes(line, "[")
    if open_bracket == -1:
        return None
    close_bracket = find_outside_quotes(line, "]", open_bracket)
    if close_bracket == -1:
        return None
    return open_bracket, close_bracket


def is_node(line: str) -> bool:
    if (indices := has_attribute_list(line)) is not None:
        before_attr = line[: indices[0]]
        return not ("->" in before_attr or before_attr.strip() == "node")
    return False


def is_edge(line: str) -> bool:
    if (indices := has_attribute_list(line)) is not None:
        return "->" in line[: indices[0]]
    return False


def try_to_parse_node(line: str) -> str | None:
    if not is_node(line):
        return None

    tokens = line.strip().split(" ")
    name: str = tokens[0]
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    return name


def try_to_parse_edge(line: str) -> tuple[str, str] | None:
    if not is_edge(line):
        return None

    # Extract source and destination endpoints
    tokens = line.strip().split(" ")
    src: str = tokens[0]
    dst: str = tokens[2]

    # Remove potential quotes around endpoints
    if src.startswith('"') and src.endswith('"'):
        src = src[1:-1]
    if dst.startswith('"') and dst.endswith('"'):
        dst = dst[1:-1]

    return src, dst


def get_attributes(line: str) -> dict[str, str]:
    # Isolate attributes from the rest of the line
    og_line = line
    indices = has_attribute_list(line)
    if indices is None:
        return {}
    line = line[indices[0] + 1 : indices[1]]

    # Parse all attributes using cursed logic
    all_attributes: dict[str, str] = {}
    while len(line) > 0:
        # Parse name
        eq_idx: int = find_outside_quotes(line, "=")
        if eq_idx == -1:
            break
        attr_name = line[:eq_idx].strip()
        line = line[eq_idx + 1 :]

        # Parse value
        attr_value: str | None = None
        for i in range(len(line)):
            if line[i] == '"':
                second_quote_idx: int = line[i + 1 :].find('"')
                if second_quote_idx == -1:
                    raise DOTParsingError(
                        f'Failed to find closing quote for value of "{attr_name}" '
                        f'attribute"',
                        og_line,
                    )
                attr_value = line[i + 1 : i + 1 + second_quote_idx]
                line = line[i + second_quote_idx + 2 :]
                break
            if str(line[i]).isalnum():
                # Find the first space, or the first comma, or the end of the line
                space_idx: int = line[i:].find(" ")
                comma_idx: int = line[i:].find(",")
                r_idx: int = len(line)
                if space_idx < comma_idx and space_idx != -1:
                    r_idx = space_idx
                elif comma_idx < space_idx and comma_idx != -1:
                    r_idx = comma_idx
                attr_value = line[i:r_idx]
                line = line[r_idx + 1 :]
                break
        if attr_value is None:
            raise DOTParsingError(
                f'Failed to parse value for attribute "{attr_name}"', og_line
            )

        # Add the attribute
        all_attributes[attr_name] = attr_value

        # Eat up the space to the next alphanumeric character, if any
        for i in range(len(line)):
            if str(line[i]).isalnum():
                line = line[i:]
                break

    return all_attributes


# Converts the input WLF file to a simpler text-based representation.
def gen_log_file(wlf_file: str, out_path: str) -> str:
    # Produce list of objects in WLF
    wlf_name: str = Path(wlf_file).stem
    obj_lst_file: str = os.path.join(out_path, f"{wlf_name}_objects.lst")
    subprocess.run(f"wlfman items -v {wlf_file} > {obj_lst_file}", shell=True)

    # Filter the list of objects to only include valid and ready signals
    obj_filter_lst_file: str = os.path.join(out_path, f"{wlf_name}_objects_filter.lst")
    with open(obj_lst_file, "r") as obj_lst_handle:
        with open(obj_filter_lst_file, "w") as obj_filter_lst_handle:
            while line := obj_lst_handle.readline():
                # Only keep output valid/ready signals
                if VALID_WIRE in line or READY_WIRE in line or DATAIN_WIRE in line:
                    obj_filter_lst_handle.write(line)

    # Produce filtered WLF file
    wlf_filter_file: str = os.path.join(out_path, f"{wlf_name}_filter.wlf")
    subprocess.run(
        f"wlfman filter -f {obj_filter_lst_file} -o {wlf_filter_file} {wlf_file}",
        shell=True,
    )

    # Produce log file
    log_file: str = os.path.join(out_path, f"{wlf_name}.log")
    subprocess.run(f"wlf2log -l duv {wlf_filter_file} > {log_file}", shell=True)

    return log_file


# Converts the .log file containing the signal changes to a CSV.
def log2csv(graph: DOTGraph, log_file: str, out_file: str) -> None:
    # Maps ID assigned to a signal in Modelsim output to the specific wire of a channel
    # it represents. Multiple IDs will map to the same channel, since there is
    # - a signal for the valid wire
    # - a signal for the ready wire
    # - as many signals as bits in the data wire (if present)
    wires: dict[int, WireReference] = {}

    # Current cycle we are at in the simulation
    cycle: int = 0

    # Maps each channel to its current state
    state: dict[Channel, ChannelState] = {}

    # Channels which had a change during the last cycle
    channels_to_update: set[Channel] = set()

    with open(os.path.join(out_file, "sim.csv"), "w") as sim:
        # Write the column names
        sim.write(
            f"cycle, src_component, src_port, dst_component, dst_port, state, data\n"
        )

        def write_updates() -> None:
            for channel in channels_to_update:
                channel_state: ChannelState = state[channel]
                valid: WireState = channel_state.valid
                ready: WireState = channel_state.ready

                # Compute dataflow state from combination of valid/ready wires
                dataflow_state: str
                if valid != WireState.LOGIC_1 and ready == WireState.LOGIC_1:
                    dataflow_state = "accept"
                elif valid == WireState.LOGIC_1 and ready != WireState.LOGIC_1:
                    dataflow_state = "stall"
                elif valid == WireState.LOGIC_1 and ready == WireState.LOGIC_1:
                    dataflow_state = "transfer"
                elif valid == WireState.LOGIC_0 and ready == WireState.LOGIC_0:
                    dataflow_state = "idle"
                else:
                    dataflow_state = "undefined"

                sim.write(
                    f"{cycle}, "
                    f"{channel.src.component}, "
                    f"{channel.src.port_id}, "
                    f"{channel.dst.component}, "
                    f"{channel.dst.port_id}, "
                    f"{dataflow_state}, "
                    f"{channel_state.data_to_dec()}\n"
                )
            channels_to_update.clear()

        # Initialize the state of each channel
        for edge in graph.edges:
            state[edge.channel] = ChannelState(edge.channel.datawidth)

        # Parse log file
        with open(log_file, "r") as log_file_handle:
            while line := log_file_handle.readline():
                tokens: list[str] = line.split(" ")
                if len(tokens) == 0:
                    break
                if tokens[0] == "D":
                    # This defines a mapping between an ID and a wire
                    wire_id = int(tokens[2])
                    wire_opt = WireReference.from_signal(graph, tokens[1])
                    if wire_opt is not None:
                        wires[wire_id] = wire_opt
                elif tokens[0] == "T":
                    # Write diff with previous state to disk
                    write_updates()

                    # This starts a new cycle (not necessarily consecutive)
                    time = int(tokens[1].replace(".", "")[:-1]) - 2000
                    if time < 0:
                        cycle = 0
                    else:
                        cycle = (time // 4000) + 1
                    pass
                elif tokens[0] == "S":
                    # This sets a signal to a specific value

                    # Just skip over wires we don't know
                    if int(tokens[1]) not in wires:
                        continue

                    # Update the corresponding channel state
                    wire_ref: WireReference = wires[int(tokens[1])]
                    wire_value: WireState = WireState.from_log(tokens[2])
                    channel_state: ChannelState = state[wire_ref.channel]
                    if wire_ref.wire_type == WireType.VALID:
                        channel_state.valid = wire_value
                    elif wire_ref.wire_type == WireType.READY:
                        channel_state.ready = wire_value
                    else:
                        assert wire_ref.data_idx is not None
                        # Just ignore accesses that ae out of bounds
                        if wire_ref.data_idx >= len(channel_state.data):
                            continue
                        idx: int = len(channel_state.data) - wire_ref.data_idx - 1
                        channel_state.data[idx] = wire_value

                    channels_to_update.add(wire_ref.channel)

            # Write diff during last state
            write_updates()


# Read the original DOT into memory and cache the sequence of edges
def parse_dot(dot_file: str) -> DOTGraph:
    graph: DOTGraph = DOTGraph(dot_file)
    all_endpoints: list[tuple[str, tuple[str, str]]] = []

    # First register all nodes...
    with open(dot_file, "r") as dot_file_handle:
        while line := dot_file_handle.readline():
            if (name := try_to_parse_node(line)) is not None:
                attributes = get_attributes(line)
                node = DOTNode(name, attributes)
                graph.nodes[name] = node
            if (endpoints := try_to_parse_edge(line)) is not None:
                all_endpoints.append((line, endpoints))

    # ...then register all edges
    for line, endpoints in all_endpoints:
        # Find the nodes the channel connects
        src_node, dst_node = endpoints
        src_og_node = src_node[1:] if src_node.startswith("_") else src_node
        dst_og_node = dst_node[1:] if dst_node.startswith("_") else dst_node

        # Find the ports the channel connects
        attributes = get_attributes(line)
        assert "from" in attributes and "to" in attributes
        src_port_id: int = int(attributes["from"][3:]) - 1
        dst_port_id: int = int(attributes["to"][2:]) - 1

        # Find the channel's bitwidth. Look for width of destination's unit
        # input port, to be consistent with the dataInArray signals we save from
        # the Modelsim waveform
        node: DOTNode = graph.nodes[dst_og_node]
        assert "in" in node.attributes
        port_str: str = node.attributes["in"].split()[dst_port_id]
        datawidth: int = int(port_str.split(":")[1].split("*")[0])

        # Append the edge to the list
        src_port = Port(src_node, src_port_id, False)
        dst_port = Port(dst_node, dst_port_id, True)
        edge = DOTEdge(
            Channel(src_port, dst_port, datawidth),
            attributes,
            src_og_node,
            dst_og_node,
        )
        graph.edges.append(edge)

        # Map both of the edge's endpoints to the channel they belong to for
        # quick access
        graph.port_to_edge[src_port] = edge
        graph.port_to_edge[dst_port] = edge
    return graph


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wlf2csv",
        description="Transform full waverform (WLF) into simpler list of events (CSV).",
    )

    parser.add_argument(
        "dot",
        metavar="dot-filepath",
        help="Path to input DOT (given to export-vhdl)",
    )
    parser.add_argument(
        "wlf",
        metavar="wlf-filepath",
        help="Path to input WLF (produced by hls-verifier)",
    )
    parser.add_argument(
        "out",
        metavar="output-path",
        help="Path to output folder",
    )
    return parser.parse_args()


def wlf2csv() -> None:
    # Parse arguments
    args = get_args()
    dot_filepath: str = args.dot
    wlf_filepath: str = args.wlf
    out_filepath: str = args.out

    # Delete output folder if it exists, then recreate it
    tmp_out_filepath: str = os.path.join(out_filepath, "tmp")
    subprocess.run(f"rm -rf {out_filepath} ; mkdir -p {tmp_out_filepath}", shell=True)

    # Parse the DOT file
    graph: DOTGraph = parse_dot(dot_filepath)

    # Pre-process the WLF file, then generate the CSV
    log_file: str = gen_log_file(wlf_filepath, tmp_out_filepath)
    log2csv(graph, log_file, out_filepath)


if __name__ == "__main__":
    wlf2csv()
