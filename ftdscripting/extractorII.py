import argparse
import csv
import os
import re
import statistics
import subprocess
from io import StringIO

KERNEL_NAME = ""
LOOP_INFO_FILE_NAME = ""
PATH_TO_OUT = ""
PATH_TO_COMP = ""
PATH_TO_SIM = ""
PATH_TO_WLF = ""
PATH_TO_CSV = ""
PATH_TO_WLF2CSV = ""
SRC_COMP = "src_component"


def get_cli_arguments():
    parser = argparse.ArgumentParser(
        prog="GSA profiling",
        description="""The script is in charge of profiling the execution 
        of an elastic circuit, getting information about the timing of the 
        multiplexers involved""",
    )

    parser.add_argument("--kernel_name", type=str, required=True)
    parser.add_argument("--component_name", type=str, required=True)
    parser.add_argument("--output_port", type=str, required=True)
    return parser.parse_args()


def set_paths(kernel_name: str):
    global KERNEL_NAME
    global LOOP_INFO_FILE_NAME
    global PATH_TO_OUT
    global PATH_TO_COMP
    global PATH_TO_SIM
    global PATH_TO_WLF
    global PATH_TO_CSV
    global PATH_TO_WLF2CSV

    KERNEL_NAME = kernel_name
    LOOP_INFO_FILE_NAME = "ftdscripting/gsaGatesInfo.txt"  # loopinfo.txt"
    PATH_TO_OUT = f"integration-test/{KERNEL_NAME}/out"
    PATH_TO_COMP = f"{PATH_TO_OUT}/comp"
    PATH_TO_SIM = f"{PATH_TO_OUT}/sim"
    PATH_TO_WLF = f"{PATH_TO_SIM}/HLS_VERIFY/vsim.wlf"
    PATH_TO_CSV = f"{PATH_TO_SIM}/HLS_VERIFY/vsim.csv"
    PATH_TO_WLF2CSV = "./bin/wlf2csv"


# The execution of a program can be analyzed using the `wlf2csv` tool, which returns a csv
# containing the information about each transaction in the trace. Out of this content,
# we need to extract all the transactions which involve a multiplexer as destination
# over input port 0 (condition port). When such a transaction is `accept`, then the condition
# token is received.
def get_sim_transactions(component_name):

    if os.path.exists(PATH_TO_CSV):
        with open(PATH_TO_CSV, "r", encoding="utf-8") as file:
            csv_result = file.read()
    else:

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

        # Write the result
        with open(f"{PATH_TO_CSV}", "w", encoding="utf-8") as file:
            file.writelines(csv_result)

    f = StringIO(csv_result)
    reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)

    # Remove usless rows from the csv
    final_csv = []
    for row in reader:
        if component_name in row[SRC_COMP]:
            final_csv.append(row)

    return final_csv


def get_sim_time(path_name):
    # Regular expression to match the desired line format
    pattern = re.compile(r"Time: (\d+) ns")

    last_time = 0

    # Open the file and read it in reverse line order
    with open(path_name, "r") as file:
        for line in reversed(file.readlines()):
            match = pattern.search(line)
            if match:
                last_time = int(match.group(1))
                break

    return last_time // 4


# Get the delay of each token at each input of each mux in the circuit
def analyze_delay_muxes(component_name, output_port, transactions):

    # Some constants
    LAST_TYPE = "last_type"
    LAST_CC = "last_time"
    DELAY = "delay"
    TRANSFER = "transfer"
    ACCEPT = "accept"
    SRC_PORT = "src_port"
    CC = "cycle"
    STATE = "state"
    TRANSACTION = "transaction"

    # Get the total simulation time
    simulation_time = get_sim_time(f"{PATH_TO_SIM}/report.txt")

    # A dictionary is used to store the information about each component.
    # The available keys are:
    #   - `last_type`: the last type of transaction, it can be either "accept", "stall" or "transfer"
    #   - `last_time`: time of the last transaction received
    #   - `number_of_transactions`: how many transactions received per port
    #   - `delay`: list of all the delays of tokens at each input port, kept in order
    data_extracted = {
        LAST_TYPE: ACCEPT,
        LAST_CC: -1,
        DELAY: [],
        TRANSACTION: [],
    }
    transactions_component = {}
    last_transaction = 0

    for transaction in transactions:
        # Store the transactions on the required port of the component
        if (
            transaction[SRC_COMP] == component_name
            and transaction[SRC_PORT] == output_port
        ):
            transactions_component[int(transaction[CC])] = transaction
            last_transaction = int(transaction[CC])

    if len(transactions_component) == 0:
        print(f"Port {output_port} of component {component_name} does not exist")
        exit(1)

    # For everys simulation cycle
    for cc in range(1, simulation_time):
        # See if there is a transaction on the clock cycle
        transaction_cc = transactions_component.get(cc)
        # Is there a transaction?
        updated = False
        delay = int(cc) - data_extracted[LAST_CC]

        # If there is a transaction
        if transaction_cc is not None:
            tt = transaction_cc[STATE]

            # If it's a transfer transaction
            if tt == TRANSFER:
                # If it is not the first transactino
                if data_extracted[LAST_CC] != -1:
                    # Add a delay between transactions
                    data_extracted[DELAY].append(delay)
                updated = True
                # Update the last cc
                data_extracted[LAST_CC] = int(cc)

            # Update the status
            data_extracted[LAST_TYPE] = tt

        # If there is not transaction but the last status is transfer, add a transaction
        elif data_extracted[LAST_TYPE] == TRANSFER and cc < last_transaction:
            data_extracted[DELAY].append(delay)
            data_extracted[LAST_TYPE] = TRANSFER
            data_extracted[LAST_CC] = int(cc)
            updated = True

        if updated:
            data_extracted[TRANSACTION].append(cc)
            print(f"Transaction @ {cc}")

    print(f"Component {component_name}; output port {output_port}:")
    print(f"\tSimulation time: {simulation_time}")
    print(f"\tNumber of transactions on port: {len(data_extracted[TRANSACTION])};")
    if len(data_extracted[TRANSACTION]) > 1:
        print(f"\tFirst transaction @ {data_extracted[TRANSACTION][0]}")
        print(f"\tLast  transaction @ {data_extracted[TRANSACTION][-1]}")
        print(
            f"\tII approx = {(data_extracted[TRANSACTION][-1] - data_extracted[TRANSACTION][0] + 1) / len(data_extracted[TRANSACTION])}"
        )


def main():

    cli_args = get_cli_arguments()
    set_paths(cli_args.kernel_name)

    # Extract the muxes transactions
    transactions = get_sim_transactions(cli_args.component_name)

    analyze_delay_muxes(cli_args.component_name, cli_args.output_port, transactions)

    return


if __name__ == "__main__":
    main()
