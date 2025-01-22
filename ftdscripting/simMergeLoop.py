from tabulate import tabulate


def getTableSimulation(isMerge: bool, inOut: int, K: int, N: int, cycles: int) -> str:

    # Sequence of tokens on the `out input` of the component
    in0_sequence = ["" for _ in range(cycles)]

    # Sequence of tokens on the `in input` of the component
    in1_sequence = ["" for _ in range(cycles)]

    # Sequence of tokens on the `outupt` of the component
    out_sequence = ["" for _ in range(cycles)]

    # Sequence of tokens from the loop itself
    loop_sequence = ["" for _ in range(cycles)]

    # Tokens arrived at the input that have not entered the loop yet, in arrival order
    inputs_to_handle = []

    # Name of the next token to handle
    token_counter = "A"

    # How many tokens are currently in the pipeline
    in_pipeline = 0

    # For each simulation cycle
    for i in range(cycles):

        # Add a new token in the input if the cycle is multiple of inOut
        if i % inOut == 0:
            in0_sequence[i] = token_counter
            inputs_to_handle.append(token_counter)
            # Get next token
            token_counter = chr(ord(token_counter) + 1)

        # If there is a token output of the loop, decrease the tokens in the pipeline
        if loop_sequence[i] != "":
            in_pipeline -= 1

        # If there is a token in the `in input` of the component, move it to the output,
        # otherwise handle a possible token from the `out input`
        # (only if `inputs_to_handle` is not empty)
        if in1_sequence[i] != "":
            out_sequence[i] = in1_sequence[i]

        elif len(inputs_to_handle) != 0:
            # Handle a new token either if we have a merge component or if the pipeline is empty
            if isMerge or in_pipeline == 0:
                new_token = inputs_to_handle[0]
                inputs_to_handle.pop(0)
                out_sequence[i] = new_token
                in_pipeline += 1
                # Move the token to each of the correct slots in the `in input` sequence
                for k in range(1, N):
                    if i + k * K < cycles:
                        in1_sequence[i + k * K] = new_token

                # Move the token to the correct `out` slot
                if i + K * N < cycles:
                    loop_sequence[i + K * N] = new_token

    # Make a table out of the simulation
    return tabulate(
        [
            ["Token outside"] + in0_sequence,
            ["Token inside"] + in1_sequence,
            ["Token out"] + out_sequence,
            ["Token loop"] + loop_sequence,
        ],
        headers=["Time"] + [str(i) for i in range(cycles)],
        tablefmt="simple_grid",
    )


def main():
    IN_OUT = 2
    K = 7
    N = 3
    CYCLES = 50
    with open("ftdscripting/simulation.txt", "w") as file:
        print(
            f"Mux   simulation: \n{getTableSimulation(False, IN_OUT, K, N, CYCLES)}\n\n",
            file=file,
        )
        print(
            f"Merge simulation: \n{getTableSimulation(True, IN_OUT, K, N, CYCLES)}\n\n",
            file=file,
        )
        print("Simulation correct; see the result in `ftdscripting/simulation.txt`")


if __name__ == "__main__":
    main()
