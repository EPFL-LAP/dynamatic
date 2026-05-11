import random

from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import Logic, LogicArray, LogicVec, LogicVecArray
from vhdl_gen.operators import BitsToOH, Op, Reduce
from vhdl_gen.configs import Configs


def _get_matrix_rank(matrix: list[list[int]]) -> int:
    """
    Computes the rank of a binary matrix over GF(2) using Gaussian elimination.

    Args:
        matrix: A list of lists representing a binary matrix (elements are 0 or 1).

    Returns:
        The rank of the matrix, which is the number of linearly independent rows.
    """

    num_rows = len(matrix)
    num_cols = len(matrix[0])
    assert num_rows > 0 and num_cols > 0, "Matrix must have at least one row and one column."

    rank = 0

    # Create a copy of the matrix to perform row operations
    mat = [row[:] for row in matrix]

    for col in range(num_cols):
        # Find a pivot row for this column
        pivot_row = None
        for row in range(rank, num_rows):
            if mat[row][col] == 1:
                pivot_row = row
                break

        if pivot_row is not None:
            # Swap the current row with the pivot row
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]

            # Eliminate below
            for row in range(rank + 1, num_rows):
                if mat[row][col] == 1:
                    # XOR the pivot row with this row
                    for c in range(col, num_cols):
                        mat[row][c] ^= mat[rank][c]

            rank += 1

    return rank


def _generate_h3_hash_matrices(num_hash_functions: int, hash_width: int, input_width: int, rng: random.Random) -> list[list[list[int]]]:
    """
    Generates a list of k H3 matrices for hardware Bloom filters.

    Dimensions:
      rows (hash_width)  = The number of output bits.
      cols (input_width) = The number of input bits.

    Args:
        num_hash_functions: Number of hash functions (matrices) to generate.
        hash_width:         The bit-width of the hash output (m).
        input_width:        The bit-width of the address input (w).
        rng:                An instance of random.Random.

    Returns:
        A list of num_hash_functions matrices of shape (hash_width x input_width), where each
        element is either 0 or 1 (with equal probability). Additionally, each matrix is guaranteed to have:
        - Full rank over GF(2)
        - No all-zero rows
        - No all-zero columns
    """
    rows = hash_width
    cols = input_width
    full_rank = min(rows, cols)

    while True:
        matrices = []
        while len(matrices) < num_hash_functions:
            matrix = [[rng.randint(0, 1) for _ in range(cols)] for _ in range(rows)]

            # 1. Check for Full Rank
            if _get_matrix_rank(matrix) < full_rank:
                continue

            # 2. Check that there are no all-zero columns
            zero_column_found = False
            for col in range(cols):
                if all(matrix[row][col] == 0 for row in range(rows)):
                    zero_column_found = True
                    break
            if zero_column_found:
                continue

            # TODO: We should also check the weight of each row/column is not too high, to avoid
            # large fan-outs/fan-ins which would deteriorate timing performance.

            # If the matrix survives all checks, add it to the list
            matrices.append(matrix)

        # 3. Check the stacked matrix has full rank
        stacked_full_rank = min(num_hash_functions * rows, cols)
        stacked_matrix = [row for matrix in matrices for row in matrix]
        if _get_matrix_rank(stacked_matrix) < stacked_full_rank:
            continue
        return matrices


class BloomFilterHash:
    def __init__(
        self,
        name: str,
        suffix: str,
        configs: Configs
    ):
        """
        BloomFilterHash

        Implements hash functions for memory addresses, for use in Bloom
        filters (approximate address comparison).

        This class encapsulates the logic for generating a VHDL module that
        computes multiple hash values for a given memory address. The output
        hash values are one-hot encoded and combined into a single bit vector,
        as used by Bloom filters. If the input address is not valid, the output
        will be all ones.

        The hash functions are drawn at random from the Carter-Wegman H_3
        family of universal hash functions. A single hash value is computed as
        a matrix-vector product of a randomly generated binary matrix and the
        input bit vector. The random binary matrix is filled with random bits
        (0 or 1) with equal probability, generated at Python level, and
        hardcoded into the VHDL module. To compute multiple hash functions,
        different random matrices are used.

        This module is fully combinational.

        NOTE: This currently does not use the Kirsch-Mitzenmacher optimization
              for > 2 hash functions. That would reduce the number of XORs, but
              it is unclear whether it is beneficial for LUT-based FPGAs, where
              XORs are cheap.

        Parameters:
            name    : Base name of the hash module.
            suffix  : Suffix appended to the entity name.
            configs : configuration generated from JSON

        Instance Variable:
            self.module_name = name + suffix : Entity and architecture identifier

        Example:
            bf_hash = BloomFilterHash(
                    name="config_0_core",
                    suffix="_bfh",
                    configs=configs
                )

            # You can later generate VHDL entity and architecture by
            #     bf_hash.generate(...)
            # You can later instantiate VHDL entity by
            #     bf_hash.instantiate(...)
        """

        self.name = name
        self.configs = configs
        self.module_name = name + suffix

        # generate hash matrices
        rng = random.Random(self.configs.bloomFilterSeed)  # fixed seed for reproducibility

        self.hash_matrices = _generate_h3_hash_matrices(
            self.configs.bloomFilterHashCount, self.configs.bloomFilterHashW, self.configs.addrW, rng)

    def generate(self, path_rtl) -> None:
        """
        Generates the VHDL 'entity' and 'architecture' sections for a Bloom filter hash module.

        Parameters:
            path_rtl    : Output directory for VHDL files.

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<self.name>.vhd.
            Entity and architecture use the identifier: <self.module_name>

        Example (BloomFilterHash):
            bfh.generate(path_rtl)

            produces in rtl/config_0_core.vhd:

            entity config_0_core_bfh is
                port(
                    ...
                );
            end entity;

            architecture arch of config_0_core_bfh is
                -- signals generated here
            begin
                -- Bloom filter hash logic here
            end architecture;

        """

        # ctx: VHDLContext for code generation state.
        # When we generate VHDL entity and architecture, we can use this context as a local variable.
        # We only need to get the context as a parameter when we instantiate the module.
        # It saves all information we need when we generate VHDL entity and architecture code.
        ctx = VHDLContext()

        ctx.tabLevel = 1
        ctx.tempCount = 0
        ctx.signalInitString = ''
        ctx.portInitString = '\tport('
        ctx.regInitString = ''
        arch = ''

        # IOs
        addr_i = LogicVec(ctx, 'addr', 'i', self.configs.addrW)
        filter_o = LogicVec(ctx, 'filter', 'o', self.configs.bloomFilterW)

        # Hashing
        hash = LogicVecArray(ctx, 'hash', 'w', self.configs.bloomFilterHashCount, self.configs.bloomFilterHashW)
        for i in range(self.configs.bloomFilterHashCount):
            for j in range(self.configs.bloomFilterHashW):
                rhs = []
                for k in range(self.configs.addrW):
                    if self.hash_matrices[i][j][k] == 1:
                        rhs.append((addr_i, k))
                        rhs.append('xor')
                if rhs:
                    rhs = rhs[:-1]  # remove trailing 'xor'
                arch += Op(ctx, (hash, i, j), *rhs)

        # One-Hot Encoding
        hash_oh = LogicVecArray(ctx, 'hash_oh', 'w', self.configs.bloomFilterHashCount, self.configs.bloomFilterW)
        for i in range(self.configs.bloomFilterHashCount):
            arch += BitsToOH(ctx, hash_oh[i], hash[i])

        # Reduce with OR to combine the one-hot encoded hashes into final filter output
        arch += Reduce(ctx, filter_o, hash_oh, 'or')

        ######   Write To File  ######
        ctx.portInitString += '\n\t);'

        # HACK: Make this compile for the moment.
        ctx.portInitString = ctx.portInitString.replace('port(;', 'port(')

        # Write to the file
        with open(f'{path_rtl}/{self.name}.vhd', 'a') as file:
            file.write('\n\n')
            file.write(ctx.library)
            file.write(f'entity {self.module_name} is\n')
            file.write(ctx.portInitString)
            file.write('\nend entity;\n\n')
            file.write(f'architecture arch of {self.module_name} is\n')
            file.write(ctx.signalInitString)
            file.write('begin\n' + arch + '\n')
            file.write('end architecture;\n')

    def instantiate(
        self,
        ctx: VHDLContext,
        instance_name: str,
        addr_i: LogicVec,
        filter_o: LogicVec
    ) -> str:
        """
        Hash Instantiation

        Creates the VHDL port mapping for the hash entity.

        Parameters:
            ctx           : VHDLContext for code generation state.
            instance_name : Identifier for this instance of the hash module (used in the label of the instantiation). Must be unique within the architecture.
            addr_i        : Input address signal to be hashed.
            filter_o      : Output signal for the Bloom filter result (one-hot encoded and OR-combined hash values).

        Returns:
            VHDL instantiation string for inclusion in the architecture body.

        Example:
            arch += hash.instantiate(
                ctx,
                addr_i    = some_address_signal,
                filter_o  = some_filter_output_signal
            )

            This generates, inside 'config_0_core.vhd' and under the 'architecture config_0_core', the following instantiation

            architecture arch of config_0_core is
                signal ...
            begin
                ...
                config_0_core_bfh : entity work.config_0_core_bfh
                    port map(
                        addr_i       => some_address_signal,
                        filter_o     => some_filter_output_signal
                    );
                ...
            end architecture;
        """

        arch = ctx.get_current_indent(
        ) + f'{instance_name} : entity work.{self.module_name}\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'port map(\n'
        ctx.tabLevel += 1

        arch += ctx.get_current_indent() + \
            f'addr_i => {addr_i.getNameRead()},\n'
        arch += ctx.get_current_indent() + \
            f'filter_o => {filter_o.getNameWrite()}\n'

        ctx.tabLevel -= 1
        arch += ctx.get_current_indent() + f');\n'
        ctx.tabLevel -= 1
        return arch
