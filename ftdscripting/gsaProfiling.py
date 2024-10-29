import csv
import re
import subprocess
from functools import cmp_to_key
from io import StringIO
from typing import List

KERNEL_NAME = "if_loop_mul"
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


class MuxesReportTable:
    def __init__(self, muxes_list: list[MuxInfo]):
        self.muxes_list: list[MuxInfo] = muxes_list
        self.table: list[str] = []
        self.compact_table: dict[str, int] = {}

    def add_row(self, row: str) -> None:
        if len(row) != len(self.muxes_list):
            raise Exception(
                f"Cannot add row {row} to table of mux due to size inconsistency"
            )
        self.table.append(row)

    def print(self) -> None:
        print([mux.mux_name for mux in self.muxes_list], sep=", ")
        print(self.compact_table)

    def build_compact_table(self) -> None:
        for row in self.table:
            if row in self.compact_table.keys():
                self.compact_table[row] += 1
            else:
                self.compact_table[row] = 1


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
        if (
            ("mux" in row["dst_component"] and "0" in row["dst_port"])
            or ("mux" in row["src_component"])
        ) and row["state"] == "accept":
            del row["src_port"]
            del row["dst_port"]
            final_csv.append(row)

    def custom_sort(a, b):
        if int(a["cycle"]) > int(b["cycle"]):
            return 1
        if int(a["cycle"]) < int(b["cycle"]):
            return -1
        else:
            if "mux" in a["dst_component"]:
                return -1
            else:
                return 1

    custom_sort_key = cmp_to_key(custom_sort)
    final_csv.sort(key=custom_sort_key)
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
                if row["dst_component"] == mux.mux_name:
                    to_keep = True
                    break

            if to_keep:
                new_table.append(mux)

        new_muxes_to_correlate.append(new_table)
    return new_muxes_to_correlate


def get_table_muxes_conditions(
    muxes_to_correlate: list[MuxInfo], transactions
) -> MuxesReportTable:

    LAST_USED = "last_index_used"
    NEXT_TO_USE = "last_index_received"
    muxes_last_value = {}
    table = MuxesReportTable(muxes_to_correlate)
    last_cycle = 0

    innermost_loop_index = muxes_to_correlate[-1].loop_depth
    muxes_in_table = [mux.mux_name for mux in muxes_to_correlate]
    muxes_in_innermost_loop = [
        mux.mux_name
        for mux in muxes_to_correlate
        if mux.loop_depth == innermost_loop_index
    ]
    updated_muxes_innermost = 0

    for tr in transactions:
        mux_name = tr["dst_component"]
        mux_name_src = tr["src_component"]
        mux_data = int(float(tr["data"]) if tr["data"] != "" else 0)
        mux_cycle = int(tr["cycle"])
        trans_has_mux_dst = "mux" in mux_name
        trans_has_mux_src = "mux" in mux_name_src

        if trans_has_mux_dst and mux_name not in muxes_in_table:
            continue

        if trans_has_mux_src and mux_name_src not in muxes_in_table:
            continue

        if mux_cycle == 0:
            if trans_has_mux_dst:
                muxes_last_value[mux_name] = {
                    LAST_USED: -1,
                    NEXT_TO_USE: mux_data,
                }
            continue

        if (
            updated_muxes_innermost == len(muxes_in_innermost_loop)
            and last_cycle != mux_cycle
        ):
            new_row = ""
            for mux in muxes_in_table:
                new_row += str(muxes_last_value[mux][LAST_USED])
            table.add_row(new_row)
            updated_muxes_innermost = 0

        if trans_has_mux_dst:
            muxes_last_value[mux_name][NEXT_TO_USE] = mux_data

        if trans_has_mux_src:
            muxes_last_value[mux_name_src][LAST_USED] = muxes_last_value[mux_name_src][
                NEXT_TO_USE
            ]
            muxes_last_value[mux_name_src][NEXT_TO_USE] = -1

            if mux_name_src in muxes_in_innermost_loop:
                updated_muxes_innermost += 1

        last_cycle = mux_cycle

    table.build_compact_table()

    return table


def main():

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
    for transaction in transactions:
        print(transaction)

    # Remove muxes which are not present in the simulation
    muxes_to_correlate = remove_unknown_muxes(muxes_to_correlate, transactions)

    # Obtain tables with values of the conditions of each set of muxes to correlate
    tables_muxes_conditions = []
    for muxes_list in muxes_to_correlate:
        tables_muxes_conditions.append(
            get_table_muxes_conditions(muxes_list, transactions)
        )

    for table in tables_muxes_conditions:
        table.print()

    return


if __name__ == "__main__":
    main()
