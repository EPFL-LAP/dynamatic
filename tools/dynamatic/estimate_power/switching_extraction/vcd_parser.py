import io
import re
import math
import bisect
import argparse
from decimal import Decimal
import time

# Handle different python versions
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

################################################################
# Environment settings
################################################################
# Define input argument parser
# arg_parser = argparse.ArgumentParser(description="Take configuration(s) from the command line")

# # Add arguments
# ## Define the path of the vcd file
# arg_parser.add_argument('-vcd_path', type=str, help="Path of the desired vcd file", default="trace.vcd")

# # Build the parser
# args = arg_parser.parse_args()

# # Get the arguments
# vcd_file = args.vcd_path

# Define regex type
RE_TYPE = type(re.compile(''))

################################################################
# Class/Function Definition
################################################################
# Define structure to store each signals separatly
class Signal(object):
    """
        Class used to store all metadata and time/value pairs of each signal in the vcd file
    """
    def __init__(self, bit_width, signal_type):
        # Define signal representations
        self.bit_width = bit_width
        self.signal_type = signal_type
        self.readable_reference = []
        self.time_value_pair_list = []
        self.endtime = None
    
    def __getitem__(self, time):
        """
            Get the signal value(s) based on the specified time

            @param time: time can be int or slice(range of time positions) 
        """
        # Check the parameter type
        if isinstance(time, slice):
            # If given a time range
            if not self.endtime:
                self.endtime = self.time_value_pair_list[-1][0]
            
            # Return the (time, value) pairs based on the given time range
            return [self[ii] for ii in range(*time.indices(self.endtime))]
        elif isinstance(time, int):
            # If given a time position
            if time < 0 :
                time = 0
            
            inserted_index = bisect.bisect_left(self.time_value_pair_list, (time, ''))

            # Define the actual timing position
            actual_timing_pos = -1

            if inserted_index == len(self.time_value_pair_list):
                # The specified timing pos is larger than the endtime of the signal
                actual_timing_pos = inserted_index - 1
            else:
                if self.time_value_pair_list[inserted_index][0] == time:
                    actual_timing_pos = inserted_index
                else:
                    actual_timing_pos = inserted_index - 1
                
            if actual_timing_pos == -1:
                return None
                
            return self.time_value_pair_list[actual_timing_pos][1]
        else:
            raise TypeError("Invalid arument type")
        
# Define event handler for potential events
class EventHandler(object):
    def enddefinitions(self, vcd_parser, signals, cur_sig_vals):
        """
            Being called when encountered $enddefinitions,
            which marks the start of value change recordings
        """
        pass

    def time(self, vcd_parser, time, cur_sig_vals):
        """
            Being called whenever a new time point is found
        """
        pass

    def value(self, vcd_parser, time, value, identifier, cur_sig_vals):
        """
            Being called when the value of a signal changes
        """
        pass

# Define Scope Class
class Scope(MutableMapping):
    def __init__(self, scope_name, vcd_parser):
        self.vcd_parser = vcd_parser
        self.scope_name = scope_name
        self.subElements = {}

    def __len__(self):
        return self.subElements.__len__()
    
    def __setitem__(self, key, value):
        return self.subElements.__setitem__(key, value)
    
    def __getitem__(self, key):
        if isinstance(key, RE_TYPE):
            pattern = '^' + re.escape(self.scope_name) + '\.' + key.pattern
            return self.vcd_parser[re.compile(pattern)]
        if key in self.subElements:
            element = self.subElements.__getitem__(key)
            if isinstance(element, Scope):
                return element

            return self.vcd_parser[element]
        
    def __delitem__(self, key):
        return self.subElements.__delitem__(key)
    
    def __iter__(self):
        return self.subElements.__iter__()
    
    def __contains__(self, __key: object) -> bool:
        return self.subElements.__contains__(__key)
    
    def __repr__(self) -> str:
        return self.scope_name +'\n{\n\t' +'\n\t'.join(self.subElements)+'\n}'
      
# Define Switching Counting class
class NodeCounting(object):
    """
        Class used to store all results of swithcing counting of a specific node
    """
    def __init__(self, node_name):
        # Define needed information
        self.node_name = node_name
        self.dataout_channel_list = []          # List used to store unique names of dataout channels of the specified node, e.x ["dataOutArray_1[0]", "dataOutArray_1[1]", ...]
        self.dataout_channel_counting = {}      # Dict used to store toggling information of different output vectors, e.x. {"dataOutArray_1" : 0}
        self.valid_channel_list = []            # List used to store unique names of valid channels of the specified node
        self.valid_channel_counting = {}        # Dict used to store toggling information of different valid channels
        self.ready_channel_list = []            # List used to store unique names of ready channels of the specified node
        self.ready_channel_counting = {}        # Dict used to store toggling information of different ready channels

        # Total switching numbers
        self.total_handshake_switching = 0
        self.total_valid_switching = 0
        self.total_ready_switching = 0
        self.total_dataout_channel_switching = 0

    def final_switching_update(self):
        # Accumulate data channel switching counting
        for value in self.dataout_channel_counting.values():
            self.total_dataout_channel_switching += value
        
        # Accumulate valid channel swithicng counting
        for value in self.valid_channel_counting.values():
            self.total_valid_switching += value

        # Accumulate ready channel switching counting
        for value in self.ready_channel_counting.values():
            self.total_ready_switching += value

        self.total_handshake_switching = self.total_ready_switching + self.total_valid_switching

    def print_switching(self):
        print("Node Name: {}".format(self.node_name))
        print("\tTotal Valid channel switching: {}".format(self.total_valid_switching))
        print("\tTotal Ready channel data switching: {}".format(self.total_ready_switching))
        print("\t[HANDSHAKE] {}".format(self.total_handshake_switching))
        print("\t[DATAOUT] {}".format(self.total_dataout_channel_switching))

    def print_switching_detail(self):
        print("Node Name: {}".format(self.node_name))
        print("\tValid Channels:")
        for key, value in self.valid_channel_counting.items():
            print("\t\t{}, {}".format(key, value))
        
        print("\tReady Channels:")
        for key, value in self.ready_channel_counting.items():
            print("\t\t{}, {}".format(key, value))

        print("\tData Channels:")
        for key, value in self.dataout_channel_counting.items():
            print("\t\t{}, {}".format(key, value))

        print("\tTotal Valid channel switching: {}".format(self.total_valid_switching))
        print("\tTotal Ready channel data switching: {}".format(self.total_ready_switching))
        print("\t[HANDSHAKE] {}".format(self.total_handshake_switching))
        print("\t[DATAOUT] {}".format(self.total_dataout_channel_switching))

# Define the vcd parser
class VcdParser(object):
    """"
        VCD_PARSER class for parsing 4-state vcd file
    """
    # Define VCD signal states
    SIG_VALUES = set((
        '0',
        '1',
        'x',
        'X',
        'z',
        'Z'
    ))

    VEC_VALUE_CHANGE = set((
        'b',
        'B',
        'r',
        'R'
    ))

    def __init__(self, 
        vcd_path = None, 
        only_signal_names = False, 
        selected_signals_list = None,
        store_time_value_pair = True,
        store_scopes = False,
        event_handler = None
    ):
        # Define inner variables
        self.hierarchy = {}             # Used to store hierarchy info of the vcd file (different scopes)
        self.scopes = {}                # Used to store information about different scopes encountered during the parsing process
        scope_stack = [self.hierarchy]
        self.signals = {}               # Dict which maps vcd's simplified signal identifier to Signal storing structure
        self.endtime = 0                # End time of the simulation
        self.begintime = 0              # Begin time of the simulation
        self.name_to_id = {}            # Dict which maps human readable signal identifiers to vcd's simplified signal identifier
        self.unique_signal_names = []   # List of unique signal identifiers
        self.timescale_dict = {}        # Dict storing info related to timescale in the vcd file
        self.signal_changed = False     # Indicate whether a signal's value has changed during at the current time or not
        self.store_data_pairs = store_time_value_pair
        self.node_switching_info = {}   # Dict used to store toggling information of different nodes in the vcd file

        # Check the input selected signal list
        if selected_signals_list is None:
            selected_signals_list = []
        
        if event_handler is None:
            event_handler = EventHandler()

        # Define status variables
        PARSE_ALL_SIGNALS = not selected_signals_list
        cur_sig_values = {}
        tmp_hier = []
        num_sigs = 0
        time = 0
        initial_flag = True

        # Define helper functions
        def handle_single_value_change(inputs):
            value = inputs[0]
            identifier = inputs[1:]
            self._add_value_change_to_signal(time, value, identifier, cur_sig_values, event_handler)

        def handle_vector_value_change(inputs):
            value, identifier = inputs[1:].split()
            self._add_value_change_to_signal(time, value, identifier, cur_sig_values, event_handler)

        # Parse the vcd file
        if vcd_path is None:
            print("ERROR -- location of the vcd file not specified!")
            exit()
        else:
            with open(vcd_path, "r") as f:
                # Parse the vcd file line by line
                while True:
                    line = f.readline()

                    # The file shall not be empty
                    if line == '':
                        break

                    # Get the keyword
                    keyword = line[0]
                    line = line.strip()

                    if line == '':
                        # empty line, skip
                        continue

                    if keyword == '#':
                        # Start of a time point, e.x. #1400
                        event_handler.time(self, time, cur_sig_values)
                        # Get the simulation time
                        time = int(line.split()[0][1:])

                        # Change status variables
                        if initial_flag:
                            self.begintime = time
                            initial_flag = False
                        self.endtime = time
                        self.signal_changed = False

                        # In case there are value change in the same line, rare...
                        value_changes = list(filter(None, line.split()[1:]))
                        if (len(value_changes) > 0):
                            for value in value_changes:
                                if value[0] in self.SIG_VALUES:
                                    handle_single_value_change(value)
                                else:
                                    # Not supported
                                    raise Exception("Same line vector value change not supported")
                    elif keyword in self.VEC_VALUE_CHANGE:
                        handle_vector_value_change(line)
                    elif keyword in self.SIG_VALUES:
                        handle_single_value_change(line)
                    elif '$enddefinitions' in line:
                        # End of signal variable definitions
                        if only_signal_names:
                            break
                        event_handler.enddefinitions(self, self.unique_signal_names, cur_sig_values)
                    elif '$scope' in line:
                        # Get the scope name
                        scope_name = line.split()[2]
                        tmp_hier.append(scope_name)

                        if store_scopes:
                            # TODO: Validate the scope handling
                            full_scope_name = '.'.join(tmp_hier)
                            new_scope = Scope(full_scope_name, self)
                            scope_stack[-1][scope_name] = new_scope
                            self.scopes[full_scope_name] = new_scope
                            scope_stack.append(new_scope) # type: ignore
                    elif '$upscope' in line:
                        tmp_hier.pop()
                        if store_scopes:
                            scope_stack.pop()
                    elif '$var' in line:
                        # Parse variable definition
                        line_split = line.split()
                        type = line_split[1]
                        size = line_split[2]
                        identifier = line_split[3]
                        tmp_name = ''.join(line_split[4:-1]) # Exclude $end
                        signal_path = '.'.join(tmp_hier)
                        
                        # Get the full human readable name of the signal
                        if signal_path:
                            signal_name = signal_path + '.' + tmp_name
                        else:
                            signal_name = tmp_name
                            
                        if store_scopes:
                            scope_stack[-1][tmp_name] = signal_name
                        
                        if (signal_name in selected_signals_list or PARSE_ALL_SIGNALS):
                            self.unique_signal_names.append(signal_name)
                            if identifier not in self.signals:
                                self.signals[identifier] = Signal(size, type)
                            self.signals[identifier].readable_reference.append(signal_name)
                            self.name_to_id[signal_name] = identifier
                            # Initialize the signal after definition
                            cur_sig_values[identifier] = 'x'
                    elif '$timescale' in line:
                        if not '$end' in line:
                            while True:
                                line += " " + f.readline().strip().rstrip()
                                if '$end' in line:
                                    break
                        timescale = ' '.join(line.split()[1:-1])
                        magnitude = Decimal(re.findall(r"\d+|$", timescale)[0])
                        
                        # Check the validity of the extracted timescale
                        if magnitude not in [1, 10, 100]:
                            print("Error: Magnitude of timescale must be one of 1, 10, or 100. "\
                                + "Current magnitude is: {}".format(magnitude))
                            exit(-1)
                        
                        unit = re.findall(r"s|ms|us|ns|ps|fs|$", timescale)[0]
                        factor = {
                            "s":  '1e0',
                            "ms": '1e-3',
                            "us": '1e-6',
                            "ns": '1e-9',
                            "ps": '1e-12',
                            "fs": '1e-15',
                        }[unit]
                        self.timescale_dict["timescale"] = magnitude * Decimal(factor)
                        self.timescale_dict["magnitude"] = magnitude
                        self.timescale_dict["unit"]   = unit
                        self.timescale_dict["factor"] = Decimal(factor)
                event_handler.time(self, time, cur_sig_values)
                for sel_sig in filter(lambda x: isinstance(x, Signal), self.signals.values()):
                    sel_sig.endtime = self.endtime       

    def _add_value_change_to_signal(self, time, value, identifier, cur_sig_vals, event_handler : EventHandler):
        # Check whether the identifier is present in the signals dict
        if identifier in self.signals:
            # Call value change handler
            event_handler.value(
                self,
                time,
                value,
                identifier,
                cur_sig_vals
            )

            # Get the corresponding storing strcuture
            selected_signal = self.signals[identifier]
            self.signal_changed = True
            if self.store_data_pairs:
                # Check whether the last value change is at the same time point
                if (len(selected_signal.time_value_pair_list) > 1 and selected_signal.time_value_pair_list[-1][0] == time):
                    #! Multiple delta may exist during simulation, we take the last value as the valid data for the signal
                    selected_signal.time_value_pair_list[-1]= (time, value)
                else:
                    selected_signal.time_value_pair_list.append((time, value))
            
            cur_sig_vals[identifier] = value

    def __getitem__(self, reg_expression):
        """
            Return matched signal or scope based on the regular expression
        """
        if isinstance(reg_expression, RE_TYPE):
            matched_signal_list = []

            for signal_name in self.unique_signal_names:
                if (reg_expression.search(signal_name)):
                    matched_signal_list.append(signal_name)
            
            for scope_name in self.scopes:
                if (reg_expression.search(scope_name)):
                    matched_signal_list.append(scope_name)
            
            if (len(matched_signal_list) == 1):
                return self[matched_signal_list[0]]
            
            return matched_signal_list
        else:
            if reg_expression in self.name_to_id:
                return self.signals[self.name_to_id[reg_expression]]
            if reg_expression in self.scopes:
                return self.scopes[reg_expression]
            
            # If not matched
            raise KeyError(reg_expression)
        
    ################################### Toggling Counting ###########################################
    def value_list_change_count(self, value_list):
        """
            This function calculates the number of transitions of the given list
            #! We IGNORE "X" -> 0 and "X" -> 1 for now
        """
        num_transitions = 0

        for i in range(len(value_list)):
            if i == 0:
                continue

            if (value_list[i][1] != value_list[i - 1][1]):
                #! Need to validate this
                if (value_list[i - 1][1] != 'x' and value_list[i - 1][1] != 'X'):
                    num_transitions += 1

        return num_transitions
    
    def get_node_handshake_toggle_count(self, node_name):
        """"
            This function calculates the handshake channel toggle counts of the specified node
            through out the whole simulation
            
            *Update_Date: 15/01/2025
        """
        
        # Create the node switching storing structure if not exist
        if (node_name not in self.node_switching_info.keys()):
            self.node_switching_info[node_name] = NodeCounting(node_name)
        
        # Define the reg pattern for valid and ready signal
        valid_signal_str = '.*' + node_name + '[^.]*_valid$'
        ready_signal_str = '.*' + node_name + '[^.]*_ready$'
        
        valid_signal_pattern = re.compile(valid_signal_str)
        ready_signal_pattern = re.compile(ready_signal_str)
        
        # Get all matched signal names
        matched_valid_signal_list = []
        matched_ready_signal_list = []
        
        for signal_name in self.unique_signal_names:
            if (valid_signal_pattern.search(signal_name)):
                matched_valid_signal_list.append(signal_name)

                # Add the name list to the final storing structure
                self.node_switching_info[node_name].valid_channel_list.append(signal_name)

            elif (ready_signal_pattern.search(signal_name)):
                matched_ready_signal_list.append(signal_name)

                # Add the name list to the final storing structure
                self.node_switching_info[node_name].ready_channel_list.append(signal_name)
                
        # Get the actual toggle count
        #! Testing
        # print("Node Name: {}".format(node_name))
        
        # Valid toggling
        # print("\tValid:")
        for valid_signal in matched_valid_signal_list:
            tmp_inner_name = self.name_to_id[valid_signal]
            tmp_value_list = self.signals[tmp_inner_name].time_value_pair_list
            tmp_toggle_number = self.value_list_change_count(tmp_value_list)
            
            # Add the toggling number to the final storing structure
            self.node_switching_info[node_name].valid_channel_counting[valid_signal] = tmp_toggle_number
            # print("\t\t{:15} {:5}".format(valid_signal.split(".")[-1], tmp_toggle_number))
            
        # Ready Toggling
        # print("\tReady:")
        for ready_signal in matched_ready_signal_list:
            tmp_inner_name = self.name_to_id[ready_signal]
            tmp_value_list = self.signals[tmp_inner_name].time_value_pair_list
            tmp_toggle_number = self.value_list_change_count(tmp_value_list)
            
            # Add the toggling number to the final storing structure
            self.node_switching_info[node_name].ready_channel_counting[ready_signal] = tmp_toggle_number
            # print("\t\t{:15} {:5}".format(ready_signal.split(".")[-1], tmp_toggle_number))
        
    
    def get_node_dataout_toggle_count(self, node_name):
        """
            This function calculates the data out channel toggle counts of the specified node
            and prints all the details
        """
        # Define the dataout channel pattern
        dataout_signal_str = r'.*' + node_name + '_(result|outs|dataOut|addrOut|dataToMem|index|trueOut|falseOut)_*\d*\['
        
        dataout_signal_pattern = re.compile(dataout_signal_str)
        
        unique_vector_name_set = set()
        
        for signal_name in self.unique_signal_names:
            if (dataout_signal_pattern.search(signal_name)):
                self.node_switching_info[node_name].dataout_channel_list.append(signal_name)
                # Remove the square bracket
                unique_vector_name_set.add(re.sub(r'\[[0-9]*\]', "", signal_name))
                
        # Get the actual dataout toggling
        for dataout_channel_name in unique_vector_name_set:
            # print("\t{}:".format(dataout_channel_name))
            tmp_total_switching = 0
            for signal_name in self.node_switching_info[node_name].dataout_channel_list: # type: ignore
                if (dataout_channel_name in signal_name):
                    tmp_inner_name = self.name_to_id[signal_name]
                    tmp_value_list = self.signals[tmp_inner_name].time_value_pair_list
                    tmp_toggle_number = self.value_list_change_count(tmp_value_list)
                    
                    #! Testing
                    # print("\t\t\t{:15} {:5}".format(signal_name, tmp_toggle_number))

                    # Sum up
                    tmp_total_switching += tmp_toggle_number
                    
            self.node_switching_info[node_name].dataout_channel_counting[dataout_channel_name] = tmp_total_switching
            # print("\t\t{:15} {:5}".format(dataout_channel_name.split(".")[-1], tmp_total_switching))
    
    def get_node_toggle_count(self, node_name):
        """
            This funciton calculates the toggle counts of both the dataout and handshake channels of the
            specified node
        """

        # Handshake channel counting
        self.get_node_handshake_toggle_count(node_name)

        # Dataout channel counting
        self.get_node_dataout_toggle_count(node_name)

        # Final update
        self.node_switching_info[node_name].final_switching_update()

    def compute_saif_bins(self, identifier):
        pairs = self.signals[identifier].time_value_pair_list
        if not pairs:
            raise ValueError(f"no value changes for {identifier}")

        # NEW: handle constant signals with exactly one recorded value
        if len(pairs) == 1:
            t, v = pairs[0]
            dur = self.endtime - self.begintime
            if v == '0':
                return dur, 0, 0, 0, 0
            elif v == '1':
                return 0, dur, 0, 0, 0
            else:
                return 0, 0, dur, 0, 0

        T0 = 0
        T1 = 0
        TX = 0

        for (t, v), (t2, _) in zip(pairs, pairs[1:]):
            dt = t2 - t
            if v == '0':
                T0 += dt
            elif v == '1':
                T1 += dt
            else:
                TX += dt

        TC = self.value_list_change_count(pairs)
        return T0, T1, TX, TC, 0
    
    def build_saif_tree(self):
        tree = {
            "tb": {
                "_nets": [],
                "_children": {
                    "duv_inst": {"_nets": [], "_children": {}}
                }
            }
        }

        duv = tree["tb"]["_children"]["duv_inst"]

        for signal_name in self.unique_signal_names:
            identifier = self.name_to_id[signal_name]
            bins = self.compute_saif_bins(identifier)

            if not signal_name.startswith("tb.duv_inst."):
                continue

            parts = signal_name.split(".")
            if len(parts) < 3:
                continue  # ignore anything not under tb.duv_inst

            # enforce fixed root structure: tb.duv_inst.<rest...>
            _, _, *rest = parts

            node = duv

            # -----------------------------
            # Case 1: net directly under duv_inst
            # -----------------------------
            if len(rest) == 1:
                netname = rest[0]
                netname = (
                    netname.replace("(", "\\(")
                        .replace(")", "\\)")
                        .replace("[", "\\[")
                        .replace("]", "\\]")
                )
                node["_nets"].append((netname, bins))
                continue

            # -----------------------------
            # Case 2: deeper hierarchy
            # -----------------------------
            *inst_path, netname = rest

            # escape net names
            netname = (
                netname.replace("(", "\\(")
                    .replace(")", "\\)")
                    .replace("[", "\\[")
                    .replace("]", "\\]")
            )

            # walk/create instance hierarchy
            for inst in inst_path:

                inst = inst.replace("(", "\\(").replace(")", "\\)")

                if inst not in node["_children"]:
                    node["_children"][inst] = {"_nets": [], "_children": {}}

                node = node["_children"][inst]

            node["_nets"].append((netname, bins))

        self.saif_tree = tree

    def write_saif_hier(self, node, name, depth=0):
        indent = "    " * depth
        out = []

        out.append(f"{indent}(INSTANCE {name}")

        # nets
        nets = node.get("_nets", [])
        if nets:
            out.append(f"{indent}    (NET")
            for netname, (T0, T1, TX, TC, IG) in nets:
                out.append(
                    f"{indent}        ({netname} (T0 {T0}) (T1 {T1}) (TX {TX}) (TC {TC}) (IG {IG}))"
                )
            out.append(f"{indent}    )")

        # children instances
        for child_name, child_node in node.get("_children", {}).items():
            out.extend(self.write_saif_hier(child_node, child_name, depth + 1))

        out.append(f"{indent})")
        return out

    def write_saif(self, out_path):
        # ensure hierarchy is built
        self.build_saif_tree()

        duration = self.endtime - self.begintime

        header = [
            "(SAIFILE",
            '    (SAIFVERSION "2.0")',
            '    (DIRECTION "backward")',
            '    (DESIGN )',
            f'    (DATE "{time.asctime()}")',
            '    (VENDOR "Mentor Graphics")',
            '    (PROGRAM_NAME "vsim")',
            '    (PROGRAM_VERSION "10.7b_1")',
            '    (DIVIDER /)',
            '    (TIMESCALE 1 fs)',
            f'    (DURATION {duration})'
        ]

        body = self.write_saif_hier(self.saif_tree["tb"], "tb", depth=1)

        with open(out_path, "w") as f:
            f.write("\n".join(header + body + [")"]))

################################################################
# Main Function
################################################################
if __name__ == '__main__':
    print("Under construction")
    