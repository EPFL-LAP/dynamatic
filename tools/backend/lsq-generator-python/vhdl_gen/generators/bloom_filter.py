import random

from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import Logic, LogicArray, LogicVec, LogicVecArray
from vhdl_gen.operators import BitsToOH, Op, Reduce
from vhdl_gen.configs import Configs


def _generate_random_h3_hash_matrix(rows: int, cols: int, rng: random.Random) -> list[list[int]]:
    """
    Generates a random binary matrix of the specified dimensions.

    Each element is either 0 or 1, with equal probability. Any returned matrix
    is guaranteed to have non-zero rows and columns.

    Parameters:
        rows : Number of rows in the matrix.
        cols : Number of columns in the matrix.
        rng  : Random number generator instance for reproducibility.
    """
    if rows == 1:
        # single row must be all ones to avoid a zero column
        return [[1] * cols]
    if cols == 1:
        # single column must be all ones to avoid a zero row
        return [[1] for _ in range(rows)]

    for _attempt in range(1000):  # retry 1000 times to avoid accidental infinite loop
        matrix = [[rng.randint(0, 1) for _ in range(cols)] for _ in range(rows)]

        # check for zero rows and columns
        retry = False
        for row in matrix:
            if all(bit == 0 for bit in row):
                retry = True
                break
        for col in range(cols):
            if all(matrix[row][col] == 0 for row in range(rows)):
                retry = True
                break
        if not retry:
            return matrix

    # TODO: This could also reject matrices which have too many ones in a row,
    #      which would lead to long XOR chains.

    assert False, "Failed to generate a valid random binary matrix after 1000 attempts. This is extremely unlikely."


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
        rng = random.Random(1)  # fixed seed for reproducibility

        self.hash_matrices = [
            _generate_random_h3_hash_matrix(self.configs.bloomFilterHashW, self.configs.addrW, rng)
            for _ in range(self.configs.bloomFilterHashCount)
        ]

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
        addr_valid_i = Logic(ctx, 'addr_valid', 'i')
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

        # Reduce with OR to combine the one-hot encoded hashes into filter output
        filter_combined = LogicVec(ctx, 'filter_combined', 'w', self.configs.bloomFilterW)
        arch += Reduce(ctx, filter_combined, hash_oh, 'or')

        # If the input address is not valid, output all ones. Otherwise, output the combined filter.
        all_ones = 2 ** self.configs.bloomFilterW - 1
        arch += Op(ctx, filter_o, filter_combined, "when", addr_valid_i, "else", all_ones)

        ######   Write To File  ######
        ctx.portInitString += '\n\t);'

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
        addr_i: LogicVec,
        addr_valid_i: Logic,
        filter_o: LogicVec
    ) -> str:
        """
        Hash Instantiation

        Creates the VHDL port mapping for the hash entity.

        Parameters:
            ctx          : VHDLContext for code generation state.
            addr_i       : Input address signal to be hashed.
            addr_valid_i : Signal indicating whether the input address is valid.
            filter_o     : Output signal for the Bloom filter result (one-hot encoded and OR-combined hash values).

        Returns:
            VHDL instantiation string for inclusion in the architecture body.

        Example:
            arch += hash.instantiate(
                ctx,
                addr_i       = some_address_signal,
                addr_valid_i = some_address_valid_signal,
                filter_o     = some_filter_output_signal
            )

            This generates, inside 'config_0_core.vhd' and under the 'architecture config_0_core', the following instantiation

            architecture arch of config_0_core is
                signal ...
            begin
                ...
                config_0_core_bfh : entity work.config_0_core_bfh
                    port map(
                        addr_i       => some_address_signal,
                        addr_valid_i => some_address_valid_signal,
                        filter_o     => some_filter_output_signal
                    );
                ...
            end architecture;
        """

        arch = ctx.get_current_indent(
        ) + f'{self.module_name} : entity work.{self.module_name}\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'port map(\n'
        ctx.tabLevel += 1

        arch += ctx.get_current_indent() + \
            f'addr_i => {addr_i.getNameRead()},\n'
        arch += ctx.get_current_indent() + \
            f'addr_valid_i => {addr_valid_i.getNameRead()},\n'
        arch += ctx.get_current_indent() + \
            f'filter_o => {filter_o.getNameWrite()}\n'

        ctx.tabLevel -= 1
        arch += ctx.get_current_indent() + f');\n'
        ctx.tabLevel -= 1
        return arch
