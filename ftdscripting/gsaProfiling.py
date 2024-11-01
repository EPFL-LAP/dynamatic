import argparse
import csv
import re
import subprocess
from io import StringIO
from typing import List

KERNEL_NAME = ""
LOOP_INFO_FILE_NAME = ""
PATH_TO_OUT = ""
PATH_TO_COMP = ""
PATH_TO_SIM = ""
PATH_TO_WLF = ""
PATH_TO_WLF2CSV = ""
DST_COMP = "dst_component"
DST_PORT = "dst_port"
CC = "cycle"
STATE = "state"


def get_cli_arguments():
    parser = argparse.ArgumentParser(
        prog="GSA profiling",
        description="""The script is in charge of profiling the execution 
        of an elastic circuit, getting information about the timing of the 
        multiplexers involved""",
    )

    parser.add_argument("--kernel_name", type=str, required=True)
    return parser.parse_args()


def set_paths(kernel_name: str):
    global KERNEL_NAME
    global LOOP_INFO_FILE_NAME
    global PATH_TO_OUT
    global PATH_TO_COMP
    global PATH_TO_SIM
    global PATH_TO_WLF
    global PATH_TO_WLF2CSV
    KERNEL_NAME = kernel_name

    LOOP_INFO_FILE_NAME = "ftdscripting/loopinfo.txt"
    PATH_TO_OUT = f"integration-test/{KERNEL_NAME}/out"
    PATH_TO_COMP = f"{PATH_TO_OUT}/comp"
    PATH_TO_SIM = f"{PATH_TO_OUT}/sim"
    PATH_TO_WLF = f"{PATH_TO_SIM}/HLS_VERIFY/vsim.wlf"
    PATH_TO_WLF2CSV = "./bin/wlf2csv"


# Information for a GSA multiplxer, including its name, type, name of the innermost loop
# it is in and its relative depth
class MuxInfo:
    def __init__(self, mux_name: str, mux_type: str, loop_name: str, loop_depth: int):
        self.mux_name: str = mux_name
        self.mux_type: str = mux_type
        self.loop_name: str = loop_name
        self.loop_depth: int = loop_depth

    def print(self):
        print(f"Mux name: {self.mux_name}")
        print(f"Mux type: {self.mux_type}")
        print(f"Loop name: {self.loop_name}")
        print(f"Loop depth: {self.loop_depth}")


# In a program, for loops are nested, froming a set of trees. This class is a node for
# a tree. Each node has a name, a depth, a set of muxes it contains and a set of
# successor nodes (representing all the nested loop)
class ForLoopNode:
    def __init__(self, name: str, depth: int):
        self.muxes: list[MuxInfo] = []
        self.nodes: list[ForLoopNode] = []
        self.depth: int = depth
        self.name: str = name

    # Add a new child to the with the information provided by `mux`
    def add_node(self, mux: MuxInfo) -> None:

        # If a node was already created for the loop `mux` is in, then
        # the mux is added to the list of that node
        for node in self.nodes:
            if node.name == mux.loop_name:
                node.add_mux(mux)
                return

        # If the depth of the loop we need to insert is immediately following the
        # one of the current node, then we add a new node to the list, containing
        # the new multiplexer
        if self.depth + 1 == mux.loop_depth:
            new_for_node = ForLoopNode(mux.loop_name, mux.loop_depth)
            new_for_node.add_mux(mux)
            self.nodes.append(new_for_node)
            return

        # Otherwise, we go over all the avilable nodes in the list and we
        # use the one which contains the current loop
        for i in range(len(self.nodes)):
            if mux.loop_name in self.nodes[i].name:
                self.nodes[i].add_node(mux)

    # Add a multiplexer to the list of muxes
    def add_mux(self, mux: MuxInfo) -> None:
        self.muxes.append(mux)

    # Print the tree
    def print(self):
        [print("| ", end="") for _ in range(self.depth - 1)]
        print(f"Loop: {self.name}")
        [print("| ", end="") for _ in range(self.depth - 1)]
        print("Muxes: ", end="")
        [print(f"{mux.mux_name} ", end="") for mux in self.muxes]
        print("")
        [node.print() for node in self.nodes]

    # Get all the possible paths from a node to its leaves in a recursive
    # fashion
    def get_muxes_traverse(self) -> List[List[MuxInfo]]:
        # Get the paths from the children to the leaves
        result: List[List[MuxInfo]] = []
        for node in self.nodes:
            result += node.get_muxes_traverse()

        # Get the muxes in the current node
        node_mux_list: List[MuxInfo] = self.muxes

        # Concatenate the muxes in the current node to what was
        # obtained from the children
        if len(result) == 0:
            result.append(node_mux_list)
        else:
            result = [node_mux_list + elem for elem in result]

        return result


# Obtain all the paths from root to leaves for all the trees in the
# set of trees
def extract_correlations(for_trees: List[ForLoopNode]) -> List[List[MuxInfo]]:
    result: list[list[MuxInfo]] = []
    for tree in for_trees:
        result += tree.get_muxes_traverse()

    return result


# Given the list of muxes, build the trees
def build_for_trees(mux_list: List) -> List[ForLoopNode]:
    mux_list.sort(key=lambda mux: mux.loop_depth)
    for_trees: list[ForLoopNode] = []

    for mux in mux_list:

        # If the depth of the mux is 1, then a new tree for that
        # for loop is to be created
        if mux.loop_depth == 1:
            found: bool = False
            for i in range(len(for_trees)):
                if for_trees[i].name == mux.loop_name:
                    found = True
                    for_trees[i].add_mux(mux)

            if not found:
                new_for_node: ForLoopNode = ForLoopNode(mux.loop_name, mux.loop_depth)
                new_for_node.add_mux(mux)
                for_trees.append(new_for_node)

        # Otherwise, we search which is the tree in which the node is
        # to be added and we insert it
        else:
            for i in range(len(for_trees)):
                if mux.loop_name in for_trees[i].name:
                    for_trees[i].add_node(mux)

    return for_trees


# Parse the file contianing information about for loops and muxes
def parse_mux_info(file_name: str) -> List[MuxInfo]:
    list_muxes: list[MuxInfo] = []

    # Open the file in read mode
    with open(file_name, "r") as loop_info_file:
        # Get all the lines (they are supposed to be in even number)
        lines: list[str] = loop_info_file.readlines()
        assert len(lines) % 2 == 0

        # Parse the data
        for mux_name_string, loop_info_string in zip(lines[0::2], lines[1::2]):
            # Get the mux name
            mux_name: str = str(re.findall(r"mux.*? ", mux_name_string)[0]).strip()
            # Get the mux type (between brackets)
            mux_type: str = re.findall(r"\((.*?)\)", mux_name_string)[0]
            # Get the loop name
            loop_name: str = re.findall(r": (.*?)$", loop_info_string)[0]
            # Remove angular branckets from the loop name
            loop_name: str = re.sub("<.*?>", "", loop_name)
            # Get loop depth
            loop_depth: int = int(re.findall(r"\d", loop_info_string)[0])
            # Add mux
            list_muxes.append(MuxInfo(mux_name, mux_type, loop_name, loop_depth))

    return list_muxes


# The execution of a program can be analyzed using the `wlf2csv` tool, which returns a csv
# containing the information about each transaction in the trace. Out of this content,
# we need to extract all the transactions which involve a multiplexer as destination
# over input port 0 (condition port). When such a transaction is `accept`, then the condition
# token is received.
def get_sim_transactions():

    # Run `wlf2csv` to obtain the execution trace
    wlf2csv_result = subprocess.run(
        [
            PATH_TO_WLF2CSV,
            f"{PATH_TO_COMP}/handshake_export.mlir",
            PATH_TO_WLF,
            KERNEL_NAME,
        ],
        capture_output=True,
        text=True,
    )

    # Get a CSV out of the execution trace
    csv_result = wlf2csv_result.stdout
    f = StringIO(csv_result)
    reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)

    # Remove usless rows from the csv
    final_csv = []
    for row in reader:
        if "mux" in row[DST_COMP]:
            final_csv.append(row)

    return final_csv


# After the `cf to handshake` pass, some muxes are removed from the simulation, and should not be taken into account.
# For this reason, we keep only the names of the multiplxers which are in the simulation
def remove_unknown_muxes(
    muxes_to_correlate: list[list[MuxInfo]], transactions
) -> list[list[MuxInfo]]:

    # New list of muxes to correlate
    new_muxes_to_correlate = []

    # For each table
    for table in muxes_to_correlate:
        new_table = []

        # For each mux
        for mux in table:

            # Keep the mux only if exists one transaction having that same mux as destination component
            to_keep = False
            for row in transactions:
                if row[DST_COMP] == mux.mux_name:
                    to_keep = True
                    break

            if to_keep:
                new_table.append(mux)

        new_muxes_to_correlate.append(new_table)
    return new_muxes_to_correlate


# Get the delay of each token at each input of each mux in the circuit
def analyze_delay_muxes(muxes_to_correlate, transactions):

    # Some constants
    LTY = "last_type"
    LTM = "last_time"
    NTR = "number_of_transactions"
    DLY = "delay"
    TRN = "transfer"
    STL = "stall"
    ACC = "accept"

    # A dictionary is used to store the information about each mux.
    # The available keys are:
    #   - `last_type`: the last type of transaction, it can be either "accept", "stall" or "transfer"
    #   - `last_time`: time of the last transaction received
    #   - `number_of_transactions`: how many transactions received per port
    #   - `delay`: list of all the delays of tokens at each input port, kept in order
    #
    # The data are available for each of the three input port, with:
    #   - `cond` being port 0;
    #   - `false` being port 1;
    #   - `true` being port 2;
    transactions_per_mux = {}
    all_muxes = []
    for table in muxes_to_correlate:
        for mux in table:
            all_muxes.append(mux.mux_name)
            # New dictionary for each mux
            transactions_per_mux[mux.mux_name] = {
                LTY: [ACC, ACC, ACC],
                LTM: [0, 0, 0],
                NTR: [0, 0, 0],
                DLY: [[], [], []],
            }

    # For each transaction
    for transaction in transactions:
        # Get the mux
        mux = transaction[DST_COMP]

        # Ignore if the mux is not to analyze
        if mux not in all_muxes:
            continue

        # Get transaction type
        tt = transaction[STATE]
        # Get transaction port
        port = int(transaction[DST_PORT])
        # Get the type of the last transaction on the same port
        last_tt = transactions_per_mux[mux][LTY][port]

        # Consider a token to be waiting if the previous transaction was `accept` and the current one
        # is either `stall` or `transfer`
        if (last_tt == ACC and tt == STL) or (last_tt == ACC and tt == TRN):
            transactions_per_mux[mux][LTM][port] = int(transaction[CC])
            transactions_per_mux[mux][LTY][port] = tt

        # Consider a token to be delivered if the previous transaction was either `stall` or `transfer`
        # and the current one is `accept`
        if (last_tt == TRN and tt == ACC) or (last_tt == STL and tt == ACC):
            # One more transaction
            transactions_per_mux[mux][NTR][port] += 1
            # Compute the delay
            delay = int(transaction[CC]) - int(transactions_per_mux[mux][LTM][port])
            # Add the delay to the list of delays, and mark the last transaction on that port as accepted
            transactions_per_mux[mux][DLY][port].append(delay)
            transactions_per_mux[mux][LTY][port] = ACC

    # TEMP: print data
    for mux, data in transactions_per_mux.items():
        print("=================================")
        print(mux)
        print(f"\tNumber Transasctions on condition input: {data[NTR][0]};")
        print(f"\tAVG delay: {sum(data[DLY][0])/data[NTR][0]}")
        print(f"\tNumber Transasctions on input 0: {data[NTR][1]};")
        print(f"\tAVG delay: {sum(data[DLY][1])/data[NTR][1]}")
        print(f"\tNumber Transasctions on input 1: {data[NTR][2]};")
        print(f"\tAVG delay: {sum(data[DLY][2])/data[NTR][2]}")
        print("=================================")


def main():

    cli_args = get_cli_arguments()
    set_paths(cli_args.kernel_name)

    # Read file with loop information
    mux_list = parse_mux_info(LOOP_INFO_FILE_NAME)

    # Build the trees of for loops
    for_trees = build_for_trees(mux_list)
    for tree in for_trees:
        tree.print()

    # Extract the list of muxes to be correlated by traversing the trees
    # from root to each leaf
    muxes_to_correlate = extract_correlations(for_trees)

    # Extract the muxes transactions
    transactions = get_sim_transactions()

    # Remove muxes which are not present in the simulation
    muxes_to_correlate = remove_unknown_muxes(muxes_to_correlate, transactions)

    analyze_delay_muxes(muxes_to_correlate, transactions)

    return


if __name__ == "__main__":
    main()
