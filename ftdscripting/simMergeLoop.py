from tabulate import tabulate


def getTableSimulation(isMerge: bool, inOut: int, K: int, N: int, cycles: int) -> str:
    in0_sequence = ["" for _ in range(cycles)]
    in1_sequence = ["" for _ in range(cycles)]
    out_sequence = ["" for _ in range(cycles)]
    loop_sequence = ["" for _ in range(cycles)]
    inputs_to_handle = []

    token_counter = "A"
    in_pipeline = 0

    for i in range(cycles):
        if i % inOut == 0:
            in0_sequence[i] = token_counter
            inputs_to_handle.append(token_counter)
            token_counter = chr(ord(token_counter) + 1)

        if loop_sequence[i] != "":
            in_pipeline -= 1

        if in1_sequence[i] != "":
            out_sequence[i] = in1_sequence[i]

        elif len(inputs_to_handle) != 0:
            if isMerge or in_pipeline == 0:
                new_token = inputs_to_handle[0]
                inputs_to_handle.pop(0)
                out_sequence[i] = new_token
                in_pipeline += 1
                for k in range(1, N):
                    if i + k * K < cycles:
                        in1_sequence[i + k * K] = new_token
                if i + K * N < cycles:
                    loop_sequence[i + K * N] = new_token

    return tabulate(
        [
            ["IN0"] + in0_sequence,
            ["IN1"] + in1_sequence,
            ["OUT"] + out_sequence,
            ["LOOP"] + loop_sequence,
        ],
        headers=["TIME"] + [str(i) for i in range(cycles)],
    )


def main():
    IN_OUT = 1
    K = 3
    N = 4
    CYCLES = 50
    print(f"Mux   simulation: \n{getTableSimulation(False, IN_OUT, K, N, CYCLES)}\n\n")
    print(f"Merge simulation: \n{getTableSimulation(True, IN_OUT, K, N, CYCLES)}\n\n")


if __name__ == "__main__":
    main()
