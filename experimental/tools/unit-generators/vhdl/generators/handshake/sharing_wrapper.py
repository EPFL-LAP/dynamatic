import re
from generators.support.logic import generate_or_n
from generators.handshake.buffers.fifo_break_none import generate_fifo_break_none
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv
from generators.handshake.lazy_fork import generate_lazy_fork


def generate_sharing_wrapper(name, params):
    bitwidth = params["bitwidth"]
    num_shared_operands = params["num_shared_operands"]
    latency = params["latency"]
    list_of_credits = params["credits"].split()

    return _generate_sharing_wrapper(name,
                                     bitwidth,
                                     num_shared_operands,
                                     latency,
                                     list_of_credits)


def _generate_sharing_wrapper(name,
                              bitwidth,
                              num_shared_operands,
                              latency,
                              list_of_credits
                              ):

    or_name = f"{name}_or"
    buff_name = f"{name}_buffer"

    group_size = len(list_of_credits)

    replication_factors = {}
    replication_factors["num_shared_operands"] = num_shared_operands
    replication_factors["group_size"] = group_size
    replication_factors["num_shared_operands_plus_1"] = num_shared_operands + 1

    or_name = f"{name}_or"
    buff_name = f"{name}_fifo_break_dv"
    lazy_fork_name = f"{name}_lazy_fork"

    dependencies = generate_or_n(or_name, {"size": group_size}) + \
        generate_fifo_break_dv(buff_name, {"num_slots": latency, "bitwidth": group_size}) + \
        generate_lazy_fork(lazy_fork_name, {"size": 2, "bitwidth": bitwidth})

    for i, num_credits in enumerate(list_of_credits):
        dependencies += generate_fifo_break_none(f"{name}_fifo_break_none_{i}", {"num_slots": num_credits, "bitwidth": bitwidth})

    entity = f"""
    library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use work.types.all;

    -- entity of sharing wrapper
    entity {name} is
      port(
        clk        : in std_logic;
        rst        : in std_logic;

    REPLICATE i:group_size
    REPLICATE j:num_shared_operands
        op[i]in[j]       : in std_logic_vector({bitwidth} - 1 downto 0);
        op[i]in[j]_valid : in std_logic;
        op[i]in[j]_ready : out std_logic;
    ENDREPLICATE j:num_shared_operands
    ENDREPLICATE i:group_size

    REPLICATE i:group_size
        op[i]out0 : out std_logic_vector({bitwidth} - 1 downto 0);
        op[i]out0_valid : out std_logic;
        op[i]out0_ready : in std_logic;
    ENDREPLICATE i:group_size

    REPLICATE i:num_shared_operands
        toSharedUnitIn[i] : out std_logic_vector({bitwidth} - 1 downto 0);
        toSharedUnitIn[i]_valid : out std_logic;
        toSharedUnitIn[i]_ready : out std_logic;
    ENDREPLICATE i:num_shared_operands

        fromSharedUnitOut0 : in std_logic_vector({bitwidth} - 1 downto 0);
        fromSharedUnitOut0_valid : in std_logic;
        fromSharedUnitOut0_ready : out std_logic
        );
    end entity;
    """

    expanded_entity = expand_replications(entity, replication_factors)

    architecture = f"""
    architecture arch of {name} is
    REPLICATE i:group_size
    REPLICATE j:num_shared_operands_plus_1
      signal sync[i]_out[j]_data : std_logic_vector({bitwidth} - 1 downto 0);
    ENDREPLICATE j:num_shared_operands_plus_1
      signal sync[i]_out0_valid : std_logic;
    ENDREPLICATE i:group_size

    REPLICATE i:num_shared_operands
      signal mux[i]_out0_data : std_logic_vector({bitwidth} - 1 downto 0);
    ENDREPLICATE i:num_shared_operands

    REPLICATE i:group_size
      signal branch0_out[i]_data : std_logic_vector({bitwidth} - 1 downto 0);
      signal branch0_out[i]_valid : std_logic;
      signal branch0_out[i]_ready : std_logic;
    ENDREPLICATE i:group_size

      signal arbiter_out : std_logic_vector({group_size} - 1 downto 0);
      signal arbiter_out_valid : std_logic;

      signal cond_buffer_out0_data : std_logic_vector({group_size} - 1 downto 0);
      signal cond_buffer_out0_valid : std_logic;
      signal cond_buffer_out0_ready : std_logic;

    REPLICATE i:group_size
      signal out_buffer[i]_out0_data : std_logic_vector({bitwidth} - 1 downto 0);
      signal out_buffer[i]_out0_valid : std_logic;
      signal out_buffer[i]_out0_ready : std_logic;
    ENDREPLICATE i:group_size


    REPLICATE i:group_size
      signal out_fork[i]_out0_data : std_logic_vector({bitwidth} - 1 downto 0);
      signal out_fork[i]_out0_valid : std_logic;
      signal out_fork[i]_out0_ready : std_logic;
      signal out_fork[i]_out1_data : std_logic_vector({bitwidth} - 1 downto 0);
      signal out_fork[i]_out1_valid : std_logic;
      signal out_fork[i]_out1_ready : std_logic;
    ENDREPLICATE i:group_size

    REPLICATE i:group_size
      signal credit[i]_out0_valid : std_logic;
      signal credit[i]_out0_ready : std_logic;
    ENDREPLICATE i:group_size


    begin
    REPLICATE i:group_size
      sync[i] : entity work.crush_sync(arch)
        generic map(
          NUM_OPERANDS => {num_shared_operands + 1},
          DATA_WIDTH   => {bitwidth}
        )
        port map(
    REPLICATE j:num_shared_operands
          ins([j]) => op[i]in[j],
    ENDREPLICATE j:num_shared_operands
          ins({num_shared_operands}) => (others => '0'),
    REPLICATE j:num_shared_operands
          ins_valid([j]) => op[i]in[j]_valid,
    ENDREPLICATE j:num_shared_operands
          ins_valid({num_shared_operands}) => credit[i]_out0_valid,
    REPLICATE j:num_shared_operands
          ins_ready([j]) => op[i]in[j]_ready,
    ENDREPLICATE j:num_shared_operands
          ins_ready({num_shared_operands}) => credit[i]_out0_ready,
    REPLICATE j:num_shared_operands_plus_1
          outs([j]) => sync[i]_out[j]_data,
    ENDREPLICATE j:num_shared_operands_plus_1
          outs_valid => sync[i]_out0_valid,
          outs_ready => arbiter_out([i])
        );

    ENDREPLICATE i:group_size

    REPLICATE i:num_shared_operands
      mux[i] : entity work.crush_oh_mux(arch)
        generic map(
          MUX_WIDTH  => {group_size},
          DATA_WIDTH => {bitwidth}
        )
        port map(
    REPLICATE j:group_size
          ins([j]) => sync[j]_out[i]_data,
    ENDREPLICATE j:group_size
          sel => arbiter_out,
          outs => mux[i]_out0_data
        );

    ENDREPLICATE i:num_shared_operands

      arbiter : entity work.bitscan(arch)
        generic map(
            SIZE => {group_size}
          )
        port map(
    REPLICATE i:group_size
          request([i]) => sync[i]_out0_valid,
    ENDREPLICATE i:group_size
          grant => arbiter_out
        );

      or_n : entity work.{or_name}(arch)
        port map(
    REPLICATE i:group_size
          ins([i]) => sync[i]_out0_valid,
    ENDREPLICATE i:group_size
          outs => arbiter_out_valid
        );

    REPLICATE i:num_shared_operands
      toSharedUnitIn[i] <= mux[i]_out0_data;
      toSharedUnitIn[i]_valid <= arbiter_out_valid;

    ENDREPLICATE i:num_shared_operands
      cond_buffer : entity work.{buff_name}(arch)
        port map(
          clk => clk,
          rst => rst,
          ins => arbiter_out,
          ins_valid => arbiter_out_valid,
          outs => cond_buffer_out0_data,
          outs_valid => cond_buffer_out0_valid,
          outs_ready => cond_buffer_out0_ready
      );

      branch : entity work.crush_oh_branch(arch)
        generic map(
          BRANCH_WIDTH => {group_size},
          DATA_WIDTH => {bitwidth}
        )
        port map(
          ins => fromSharedUnitOut0,
          ins_valid => fromSharedUnitOut0_valid,
          ins_ready => fromSharedUnitOut0_ready,
REPLICATE i:group_size
          outs([i]) => branch0_out[i]_data,
ENDREPLICATE i:group_size
REPLICATE i:group_size
          outs_valid([i]) => branch0_out[i]_valid,
ENDREPLICATE i:group_size
REPLICATE i:group_size
          outs_ready([i]) => branch0_out[i]_ready,
ENDREPLICATE i:group_size
          sel => cond_buffer_out0_data,
          sel_valid => cond_buffer_out0_valid,
          sel_ready => cond_buffer_out0_ready
        );

    REPLICATE i:group_size
      out_buffer[i] : entity work.{name}_fifo_break_none_[i](arch)
        port map(
          clk => clk,
          rst => rst,
          ins => branch0_out[i]_data,
          ins_valid => branch0_out[i]_valid,
          ins_ready => branch0_out[i]_ready,
          outs => out_buffer[i]_out0_data,
          outs_valid => out_buffer[i]_out0_valid,
          outs_ready => out_buffer[i]_out0_ready
        );

    ENDREPLICATE i:group_size
    REPLICATE i:group_size
      out_fork[i] : entity work.{lazy_fork_name}(arch)
        port map(
          clk => clk,
          rst => rst,
          ins => out_buffer[i]_out0_data,
          ins_valid => out_buffer[i]_out0_valid,
          ins_ready => out_buffer[i]_out0_ready,
          outs(0) => out_fork[i]_out0_data,
          outs(1) => out_fork[i]_out1_data,
          outs_valid(0) => out_fork[i]_out0_valid,
          outs_valid(1) => out_fork[i]_out1_valid,
          outs_ready(0) => op[i]out0_ready,
          outs_ready(1) => out_fork[i]_out1_ready
        );

    ENDREPLICATE i:group_size
    REPLICATE i:group_size
      credit[i] : entity work.crush_credit_dataless(arch)
        generic map(
          NUM_CREDITS => [list_of_credits[i]]
        )
        port map(
          clk => clk,
          rst => rst,
          ins_valid => out_fork[i]_out1_valid,
          ins_ready => out_fork[i]_out1_ready,
          outs_valid => credit[i]_out0_valid,
          outs_ready => credit[i]_out0_ready
        );

    ENDREPLICATE i:group_size
    REPLICATE i:group_size
      op[i]out0 <= out_fork[i]_out0_data;
      op[i]out0_valid <= out_fork[i]_out0_valid;

    ENDREPLICATE i:group_size
    end architecture;
    """

    # match and replace "[list_of_credits[replication_variable]]"
    # with the corresponding value of list_of_credits
    # allowing pre-replication value insertion
    lists = {
        "list_of_credits": list_of_credits
    }

    expanded_architecture = expand_replications(architecture, replication_factors, lists)

    return dependencies + expanded_entity + expanded_architecture


# Take an f-string and perform parameterized replication
def expand_replications(s, replication_factors, lists={}):
    # this is to allow different values per replication
    # it matches [listname[replication_variable]]
    # and then for the n-th replication,
    # inserts the n-th value in that list
    def substitute_list_indexing(line, idx_var, j):
        line = re.sub(
            rf"\[([a-zA-Z_]\w*)\[{idx_var}\]\]",
            lambda m: str(lists[m.group(1)][j]),
            line
        )
        return line

    lines = s.splitlines()
    while True:
        new_lines = []
        i = 0
        changed = False
        while i < len(lines):
            line = lines[i]
            match = re.match(r'REPLICATE (\w+):(\w+)', line.strip())
            if match:
                idx_var, count_key = match.groups()
                count = replication_factors[count_key]
                block = []
                i += 1
                while not re.match(rf'ENDREPLICATE {idx_var}:{count_key}', lines[i].strip()):
                    block.append(lines[i])
                    i += 1
                i += 1  # skip ENDREPLICATE
                for j in range(count):
                    for block_line in block:
                        replaced = substitute_list_indexing(block_line, idx_var, j)
                        replaced = replaced.replace(f"[{idx_var}]", str(j))
                        new_lines.append(replaced)
                changed = True
            else:
                new_lines.append(line)
                i += 1
        lines = new_lines
        if not changed:
            break
    return "\n".join(lines)
