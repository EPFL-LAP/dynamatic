import re
from typing import List

LOOP_INFO_FILE_NAME = "ftdscripting/loopinfo.txt"


# Information for a GSA multiplxer, including its name, type, name of the innermost loop
# it is in and its relative depth
class MuxInfo:
    def __init__(self, mux_name: str, mux_type: str, loop_name: str, loop_depth: int):
        self.mux_name = mux_name
        self.mux_type = mux_type
        self.loop_name = loop_name
        self.loop_depth = loop_depth

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
        self.muxes = []
        self.nodes = []
        self.depth = depth
        self.name = name

    # Add a new child to the with the information provided by `mux`
    def add_node(self, mux: MuxInfo):

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
    def add_mux(self, mux: MuxInfo):
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
    def get_muxes_traverse(self) -> List[List[str]]:
        # Get the paths from the children to the leaves
        result = []
        for node in self.nodes:
            result += node.get_muxes_traverse()

        # Get the muxes in the current node
        node_mux_list = [mux.mux_name for mux in self.muxes]

        # Concatenate the muxes in the current node to what was
        # obtained from the children
        if len(result) == 0:
            result.append(node_mux_list)
        else:
            result = [node_mux_list + elem for elem in result]

        return result


# Obtain all the paths from root to leaves for all the trees in the
# set of trees
def extract_correlations(for_trees: List[ForLoopNode]) -> List[List[str]]:
    result = []
    for tree in for_trees:
        result += tree.get_muxes_traverse()

    return result


# Given the list of muxes, build the trees
def build_for_trees(mux_list: List) -> List[ForLoopNode]:
    mux_list.sort(key=lambda mux: mux.loop_depth)
    for_trees = []

    for mux in mux_list:

        # If the depth of the mux is 1, then a new tree for that
        # for loop is to be created
        if mux.loop_depth == 1:
            found = False
            for i in range(len(for_trees)):
                if for_trees[i].name == mux.loop_name:
                    found = True
                    for_trees[i].add_mux(mux)

            if not found:
                new_for_node = ForLoopNode(mux.loop_name, mux.loop_depth)
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
def parse_mux_info(file_name) -> List[MuxInfo]:
    list_muxes = []

    # Open the file in read mode
    with open(file_name, "r") as loop_info_file:
        # Get all the lines (they are supposed to be in even number)
        lines = loop_info_file.readlines()
        assert len(lines) % 2 == 0

        # Parse the data
        for mux_name_string, loop_info_string in zip(lines[0::2], lines[1::2]):
            # Get the mux name
            mux_name = re.findall(r"mux.*? ", mux_name_string)[0]
            # Get the mux type (between brackets)
            mux_type = re.findall(r"\((.*?)\)", mux_name_string)[0]
            # Get the loop name
            loop_name = re.findall(r": (.*?)$", loop_info_string)[0]
            # Remove angular branckets from the loop name
            loop_name = re.sub("<.*?>", "", loop_name)
            # Get loop depth
            loop_depth = int(re.findall(r"\d", loop_info_string)[0])
            # Add mux
            list_muxes.append(MuxInfo(mux_name, mux_type, loop_name, loop_depth))

    return list_muxes


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
    print(muxes_to_correlate)
    print(f"Number of tables to generate: {len(muxes_to_correlate)}")

    return


if __name__ == "__main__":
    main()
