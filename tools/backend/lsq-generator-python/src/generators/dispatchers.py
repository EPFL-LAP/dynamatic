from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import *
from vhdl_gen.operators import *


class PortToQueueDispatcher:
    def __init__(
        self,
        name: str,
        suffix: str,
        numPorts: int,
        numEntries: int,
        bitsW: int,
        portAddrW: int
    ):
        """
        Port-to-Queue (Port-to-Entry) Dispatcher

        Models a dispatcher that routes signals from multiple ports to queue entries.

        This class encapsulates the logic for generating a VHDL module that takes
        arguments from a specific access port and passes them to a corresponding
        queue entry. 

        This generates three main parts in the LSQ module:
            1. Load Address Port Dispatcher
            2. Store Address Port Dispatcher
            3. Store Data Port Dispatcher

        Initilization Parameters:
            name       : Base name of the dispatcher.
            suffix     : Suffix appended to the entity name.
                - lda: Load Address Port Dispatcher
                - sta: Store Address Port Dispatcher
                - std: Store Data Port Dispatcher
            numPorts   : Number of access ports.
            numEntries : Number of queue entries.
            bitsW      : Width of each data/address bus.
            portAddrW  : Width of the port index bus.

        Instance Variable:
            self.module_name = name + suffix : Entity and architecture identifier

        Example (Load Address Port Dispatcher):
            ptq_dispatcher_lda = PortToQueueDispatcher(
                                    "config_0_core",
                                    "_lda", 
                                    configs.numLdPorts, 
                                    configs.numLdqEntries, 
                                    configs.addrW, 
                                    configs.ldpAddrW
                                )

            # You can later generate VHDL entity and architecture by
            #     ptq_dispatcher_lda.generate(...)
            # You can later instantiate VHDL entity by
            #     ptq_dispatcher_lda.instantiate(...)

        """

        self.name = name
        self.module_name = name + suffix
        self.numPorts = numPorts
        self.numEntries = numEntries
        self.bitsW = bitsW
        self.portAddrW = portAddrW

    def generate(self, path_rtl) -> None:
        """
        Generates the VHDL 'entity' and 'architecture' sections for a dispatcher
        that passes arguments from a specific access port to a corresponding queue entry.

        Parameters:
            path_rtl    : Output directory for VHDL files.

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<self.name>.vhd.
            Entity and architecture use the identifier: <self.module_name>

        Example (Load Address Port Dispatcher):
            ptq_dispatcher_lda.generate(path_rtl="rtl")

            produces in rtl/config_0_core.vhd:

            entity config_0_core_lda is
                port(
                    rst           : in  std_logic;
                    clk           : in  std_logic;
                    ...
                );
            end entity;

            architecture arch of config_0_core_lda is
                -- signals generated here
            begin
                -- dispatcher logic here
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
        ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        arch = ''

        # IOs
        port_payload_i = LogicVecArray(
            ctx, 'port_payload', 'i', self.numPorts, self.bitsW)
        port_valid_i = LogicArray(ctx, 'port_valid', 'i', self.numPorts)
        port_ready_o = LogicArray(ctx, 'port_ready', 'o', self.numPorts)
        entry_alloc_i = LogicArray(ctx, 'entry_alloc', 'i', self.numEntries)
        entry_payload_valid_i = LogicArray(
            ctx, 'entry_payload_valid', 'i', self.numEntries)
        if (self.numPorts != 1):
            entry_port_idx_i = LogicVecArray(
                ctx, 'entry_port_idx', 'i', self.numEntries, self.portAddrW)
        entry_payload_o = LogicVecArray(
            ctx, 'entry_payload', 'o', self.numEntries, self.bitsW)
        entry_wen_o = LogicArray(ctx, 'entry_wen', 'o', self.numEntries)
        queue_head_oh_i = LogicVec(ctx, 'queue_head_oh', 'i', self.numEntries)

        # one-hot port index
        entry_port_idx_oh = LogicVecArray(
            ctx, 'entry_port_idx_oh', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            if (self.numPorts == 1):
                arch += Op(ctx, entry_port_idx_oh[i], 1)
            else:
                arch += BitsToOH(ctx, entry_port_idx_oh[i], entry_port_idx_i[i])

        # Mux for the data/addr
        for i in range(0, self.numEntries):
            arch += Mux1H(ctx, entry_payload_o[i],
                          port_payload_i, entry_port_idx_oh[i])

        # Entries that request data/address from a any port
        entry_ptq_ready = LogicArray(
            ctx, 'entry_ptq_ready', 'w', self.numEntries)
        for i in range(0, self.numEntries):
            arch += Op(ctx, entry_ptq_ready[i], entry_alloc_i[i],
                       'and', 'not', entry_payload_valid_i[i])

        # Entry-port pairs that the entry request the data/address from the port
        entry_waiting_for_port = LogicVecArray(
            ctx, 'entry_waiting_for_port', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            arch += Op(ctx, entry_waiting_for_port[i], entry_port_idx_oh[i],
                       'when', entry_ptq_ready[i], 'else', 0)

        # Reduce the matrix for each entry to get the ready signal:
        # If one or more entries is requesting data/address from a certain port, ready is set high.
        port_ready_vec = LogicVec(ctx, 'port_ready_vec', 'w', self.numPorts)
        arch += Reduce(ctx, port_ready_vec, entry_waiting_for_port, 'or')
        arch += VecToArray(ctx, port_ready_o, port_ready_vec)

        # AND the request signal with valid, it shows entry-port pairs that are both valid and ready.
        entry_port_options = LogicVecArray(
            ctx, 'entry_port_options', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            for j in range(0, self.numPorts):
                arch += ctx.get_current_indent() + f'{entry_port_options.getNameWrite(i, j)} <= ' \
                    f'{entry_waiting_for_port.getNameRead(i, j)} and {port_valid_i.getNameRead(j)};\n'

        # For each port, the oldest entry receives bit this cycle. The priority masking per port(column)
        # generates entry-port pairs that will tranfer data/address this cycle.
        entry_port_transfer = LogicVecArray(
            ctx, 'entry_port_transfer', 'w', self.numEntries, self.numPorts)
        arch += CyclicPriorityMasking(ctx, entry_port_transfer,
                                      entry_port_options, queue_head_oh_i)

        # Reduce for each entry(row), which generates write enable signal for entries
        for i in range(0, self.numEntries):
            arch += Reduce(ctx, entry_wen_o[i], entry_port_transfer[i], 'or')

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
        ctx:                VHDLContext,
        port_payload_i:        LogicVecArray,
        port_valid_i:       LogicArray,
        port_ready_o:       LogicArray,
        entry_alloc_i:      LogicArray,
        entry_payload_valid_i: LogicArray,
        entry_port_idx_i:   LogicVecArray,
        entry_payload_o:       LogicVecArray,
        entry_wen_o:        LogicArray,
        queue_head_oh_i:    LogicVec
    ) -> str:
        """
        Port-to-Queue Dispatcher Instantiation

        Creates the VHDL port mapping for the Port-to-Queue dispatcher entity.
        Connects the top-level signals (reset, clock, port and entry signals)
        to the internal dispatcher instance named <self.module_name>_dispatcher.

        Parameters:
            ctx                  : VHDLContext for code generation state.
            port_payload_i          : Input data or address bits from each port
            port_valid_i         : Valid signal for each input port (Valid data/address)
            port_ready_o         : Ready signal indicating the queue is ready to receive data/address
            entry_alloc_i        : Allocation bit for a queue entry
            entry_payload_valid_i: Valid bit for the data/address of a queue entry
            entry_port_idx_i     : Indicates to which port the entry is assigned
            entry_payload_o         : Output bits written to the entry
            entry_wen_o          : Write enable for each entry 
            queue_head_oh_i      : One-hot vector indicating the current head index of the queue.

        Returns:
            VHDL instantiation string for inclusion in the architecture body.

        Example (Load Address Port Dispatcher):
            arch += ptq_dispatcher_lda.instantiate(
                ctx,
                port_payload_i         = ldp_addr_i,
                port_valid_i        = ldp_addr_valid_i,
                port_ready_o        = ldp_addr_ready_o,
                entry_alloc_i       = ldq_valid,
                entry_payload_valid_i  = ldq_addr_valid,
                entry_port_idx_i    = ldq_port_idx,
                entry_payload_o        = ldq_addr,
                entry_wen_o         = ldq_addr_wen,
                queue_head_oh_i     = ldq_head_oh
            )

            This generates, inside 'config_0_core.vhd' and under the 'architecture config_0_core', the following instantiation

            architecture arch of config_0_core is
                signal ...
            begin
                ...

                config_0_core_lda_dispatcher : entity work.config_0_core_lda
                    port map(
                        rst => rst,
                        clk => clk,
                        port_payload_0_i => ldp_addr_0_i,
                        port_payload_1_i => ldp_addr_1_i,
                        port_ready_0_o => ldp_addr_ready_0_o,
                        port_ready_1_o => ldp_addr_ready_1_o,
                        port_valid_0_i => ldp_addr_valid_0_i,
                        port_valid_1_i => ldp_addr_valid_1_i,
                        entry_alloc_0_i => ldq_valid_0_q,
                        entry_alloc_1_i => ldq_valid_1_q,
                        entry_payload_valid_0_i => ldq_addr_valid_0_q,
                        entry_payload_valid_1_i => ldq_addr_valid_1_q,
                        entry_port_idx_0_i => ldq_port_idx_0_q,
                        entry_port_idx_1_i => ldq_port_idx_1_q,
                        entry_payload_0_o => ldq_addr_0_d,
                        entry_payload_1_o => ldq_addr_1_d,
                        entry_wen_0_o => ldq_addr_wen_0,
                        entry_wen_1_o => ldq_addr_wen_1,
                        queue_head_oh_i => ldq_head_oh
                    );
                ...
            end architecture;

        """

        arch = ctx.get_current_indent(
        ) + f'{self.module_name}_dispatcher : entity work.{self.module_name}\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'port map(\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'rst => rst,\n'
        arch += ctx.get_current_indent() + f'clk => clk,\n'
        for i in range(0, self.numPorts):
            arch += ctx.get_current_indent() + \
                f'port_payload_{i}_i => {port_payload_i.getNameRead(i)},\n'
        for i in range(0, self.numPorts):
            arch += ctx.get_current_indent() + \
                f'port_ready_{i}_o => {port_ready_o.getNameWrite(i)},\n'
        for i in range(0, self.numPorts):
            arch += ctx.get_current_indent() + \
                f'port_valid_{i}_i => {port_valid_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_alloc_{i}_i => {entry_alloc_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_payload_valid_{i}_i => {entry_payload_valid_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            if (self.numPorts != 1):
                arch += ctx.get_current_indent() + \
                    f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_payload_{i}_o => {entry_payload_o.getNameWrite(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_wen_{i}_o => {entry_wen_o.getNameWrite(i)},\n'
        arch += ctx.get_current_indent() + \
            f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
        ctx.tabLevel -= 1
        arch += ctx.get_current_indent() + f');\n'
        ctx.tabLevel -= 1
        return arch


class QueueToPortDispatcher:
    def __init__(
        self,
        name: str,
        suffix: str,
        numPorts: int,
        numEntries: int,
        bitsW: int,
        portAddrW: int
    ):
        """
        Queue-to-Port (Entry-to-Port) Dispatcher

        Models a dispatcher that routes signals from queue entries to access ports.

        This class encapsulates the logic for generating a VHDL module that takes
        data from queue entries and routes it to the correct outgoing port based on
        priority. 

        This generates one main part in the LSQ module:
            1. Load Data Port Dispatcher
            2. (Optionally) Store Backward Port Dispatcher

        Initialization Parameters:
            name        : Base name of the dispatcher.
            suffix      : Suffix appended to the entity name.
            numPorts    : Number of access ports.
            numEntries  : Number of queue entries.
            bitsW       : Width of each data bus.
            portAddrW   : Width of the port index bus.

        Instance Variable:
            self.module_name = name + suffix : Entity and architecture identifier

        Example (Load Data Port Dispatcher):
            qtp_dispatcher_ldd = QueueToPortDispatcher(
                                    name="config_0_core",
                                    suffix="_ldd",
                                    numPorts=configs.numLdPorts,
                                    numEntries=configs.numLdqEntries,
                                    bitsW=configs.addrW,
                                    portAddrW=configs.ldpAddrW
                                )

            # You can later generate VHDL entity and architecture by
            #     qtp_dispatcher_ldd.generate(...)
            # You can later instantiate VHDL entity by
            #     qtp_dispatcher_ldd.instantiate(...)
        """

        self.name = name
        self.module_name = name + suffix

        self.numPorts = numPorts
        self.numEntries = numEntries
        self.bitsW = bitsW
        self.portAddrW = portAddrW

    def generate(self, path_rtl) -> None:
        """
        Queue-to-Port (Entry-to-Port) Dispatcher

        Generates the VHDL 'entity' and 'architecture' sections for a dispatcher
        that routes data from queue entries to their access ports.

        Parameters:
            path_rtl    : Output directory for VHDL files.

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<self.name>_core.vhd.
            Entity and architecture use the identifier: <self.module_name>

        Example (Load Data Port Dispatcher):
            qtp_dispatcher_ldd.generate(path_rtl="rtl")

            produces in rtl/config_0_core.vhd:

            entity config_0_core_ldd is
                port(
                    rst           : in  std_logic;
                    clk           : in  std_logic;
                    ...
                );
            end entity;

            architecture arch of config_0_core_ldd is
                -- signals generated here
            begin
                -- dispatcher logic here
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
        ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        arch = ''

        # IOs
        if (self.bitsW != 0):
            port_payload_o = LogicVecArray(
                ctx, 'port_payload', 'o', self.numPorts, self.bitsW)
        port_valid_o = LogicArray(ctx, 'port_valid', 'o', self.numPorts)
        port_ready_i = LogicArray(ctx, 'port_ready', 'i', self.numPorts)
        entry_alloc_i = LogicArray(ctx, 'entry_alloc', 'i', self.numEntries)
        entry_payload_valid_i = LogicArray(
            ctx, 'entry_payload_valid', 'i', self.numEntries)
        if (self.numPorts != 1):
            entry_port_idx_i = LogicVecArray(
                ctx, 'entry_port_idx', 'i', self.numEntries, self.portAddrW)
        if (self.bitsW != 0):
            entry_payload_i = LogicVecArray(
                ctx, 'entry_payload', 'i', self.numEntries, self.bitsW)
        entry_reset_o = LogicArray(ctx, 'entry_reset', 'o', self.numEntries)
        queue_head_oh_i = LogicVec(ctx, 'queue_head_oh', 'i', self.numEntries)

        # one-hot port index
        entry_port_idx_oh = LogicVecArray(
            ctx, 'entry_port_idx_oh', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            if (self.numPorts == 1):
                arch += Op(ctx, entry_port_idx_oh[i], 1)
            else:
                arch += BitsToOH(ctx, entry_port_idx_oh[i], entry_port_idx_i[i])

        # This matrix shows entry-port pairs that the entry is linked with the port
        entry_allocated_for_port = LogicVecArray(
            ctx, 'entry_allocated_for_port', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            arch += Op(ctx, entry_allocated_for_port[i], entry_port_idx_oh[i],
                       'when', entry_alloc_i[i], 'else', 0)

        # For each port, the oldest entry send bits this cycle. The priority masking per port(column)
        # generates entry-port pairs that will tranfer data/address this cycle.
        # It is also used as one-hot select signal for data Mux.
        oldest_entry_allocated_per_port = LogicVecArray(
            ctx, 'oldest_entry_allocated_per_port', 'w', self.numEntries, self.numPorts)
        arch += CyclicPriorityMasking(ctx, oldest_entry_allocated_per_port,
                                      entry_allocated_for_port, queue_head_oh_i)

        if (self.bitsW != 0):
            for j in range(0, self.numPorts):
                arch += Mux1H(ctx, port_payload_o[j],
                              entry_payload_i, oldest_entry_allocated_per_port, j)

        # Mask the matrix with dataValid
        entry_waiting_for_port_valid = LogicVecArray(
            ctx, 'entry_waiting_for_port_valid', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            arch += Op(ctx, entry_waiting_for_port_valid[i], oldest_entry_allocated_per_port[i],
                       'when', entry_payload_valid_i[i], 'else', 0)

        # Reduce the matrix for each port to get the valid signal:
        # If an entry is providing data/address from a certain port, valid is set high.
        port_valid_vec = LogicVec(ctx, 'port_valid_vec', 'w', self.numPorts)
        arch += Reduce(ctx, port_valid_vec, entry_waiting_for_port_valid, 'or')
        arch += VecToArray(ctx, port_valid_o, port_valid_vec)

        # AND the request signal with ready, it shows entry-port pairs that are both valid and ready.
        entry_port_transfer = LogicVecArray(
            ctx, 'entry_port_transfer', 'w', self.numEntries, self.numPorts)
        for i in range(0, self.numEntries):
            for j in range(0, self.numPorts):
                arch += ctx.get_current_indent() + f'{entry_port_transfer.getNameWrite(i, j)} <= ' \
                    f'{entry_waiting_for_port_valid.getNameRead(i, j)} and {port_ready_i.getNameRead(j)};\n'

        # Reduce for each entry(row), which generates reset signal for entries
        for i in range(0, self.numEntries):
            arch += Reduce(ctx, entry_reset_o[i], entry_port_transfer[i], 'or')

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
        ctx:                VHDLContext,
        port_payload_o:        LogicVecArray,
        port_valid_o:       LogicArray,
        port_ready_i:       LogicArray,
        entry_alloc_i:      LogicArray,
        entry_payload_valid_i: LogicArray,
        entry_port_idx_i:   LogicVecArray,
        entry_payload_i:       LogicVecArray,
        entry_reset_o:      LogicArray,
        queue_head_oh_i:    LogicVec
    ) -> str:
        """
        Queue-to-Port Dispatcher Instantiation

        Creates the VHDL port mapping for the Queue-to-Port dispatcher entity.
        Connects the top-level signals (reset, clock, entry and port signals)
        to the internal dispatcher instance named <self.module_name>_dispatcher.

        Parameters:
            ctx                  : VHDLContext for code generation state.
            port_payload_o          : Output data bits from each queue entry
            port_valid_o         : Valid signal for each input port (Valid data)
            port_ready_i         : Ready signal indicating the queue is ready to send data
            entry_alloc_i        : Valid bit for a queue entry
            entry_payload_valid_i   : Valid bit for the contents of a queue entry
            entry_port_idx_i     : Indicates to which port the entry is assigned
            entry_payload_i         : Input data bits which is written in the queue entry
            entry_reset_o        : Array of reset outputs for entries.
            queue_head_oh_i      : One-hot vector indicating the current head index of the queue.

        Returns:
            VHDL instantiation string for inclusion in the architecture body.


        Example:
            # Base architecture: 'config_0_core'
            # suffix for Load Data Dispatcher instantiation: '_ldd'

            arch += qtp_dispatcher_ldd.instantiate(
                ctx,
                port_payload_o         = ldp_data_o,
                port_valid_o        = ldp_data_valid_o,
                port_ready_i        = ldp_data_ready_i,
                entry_alloc_i       = ldq_valid,
                entry_payload_valid_i  = ldq_data_valid,
                entry_port_idx_i    = ldq_port_idx,
                entry_payload_i        = ldq_data,
                entry_reset_o       = ldq_reset,
                queue_head_oh_i     = ldq_head_oh
            )

            This generates, inside 'config_0_core.vhd' and under the 'architecture config_0_core', the following instantiation

            architecture arch of config_0_core is
                signal ...
            begin
                ...
                config_0_core_ldd_dispatcher : entity work.config_0_core_ldd
                    port map(
                        rst => rst,
                        clk => clk,
                        port_payload_0_o => ldp_data_0_o,
                        port_payload_1_o => ldp_data_1_o,
                        port_ready_0_i => ldp_data_ready_0_i,
                        port_ready_1_i => ldp_data_ready_1_i,
                        port_valid_0_o => ldp_data_valid_0_o,
                        port_valid_1_o => ldp_data_valid_1_o,
                        entry_alloc_0_i => ldq_valid_0_q,
                        entry_alloc_1_i => ldq_valid_1_q,
                        entry_payload_valid_0_i => ldq_data_valid_0_q,
                        entry_payload_valid_1_i => ldq_data_valid_1_q,
                        entry_port_idx_0_i => ldq_port_idx_0_q,
                        entry_port_idx_1_i => ldq_port_idx_1_q,
                        entry_payload_0_i => ldq_data_0_q,
                        entry_payload_1_i => ldq_data_1_q,
                        entry_reset_0_o => ldq_reset_0,
                        entry_reset_1_o => ldq_reset_1,
                        queue_head_oh_i => ldq_head_oh
                    );
                ...
            end architecture;
        """

        arch = ctx.get_current_indent(
        ) + f'{self.module_name}_dispatcher : entity work.{self.module_name}\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'port map(\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'rst => rst,\n'
        arch += ctx.get_current_indent() + f'clk => clk,\n'
        for i in range(0, self.numPorts):
            if (port_payload_o != None):
                arch += ctx.get_current_indent() + \
                    f'port_payload_{i}_o => {port_payload_o.getNameWrite(i)},\n'
        for i in range(0, self.numPorts):
            arch += ctx.get_current_indent() + \
                f'port_ready_{i}_i => {port_ready_i.getNameRead(i)},\n'
        for i in range(0, self.numPorts):
            arch += ctx.get_current_indent() + \
                f'port_valid_{i}_o => {port_valid_o.getNameWrite(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_alloc_{i}_i => {entry_alloc_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_payload_valid_{i}_i => {entry_payload_valid_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            if (self.numPorts != 1):
                arch += ctx.get_current_indent() + \
                    f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            if (entry_payload_i != None):
                arch += ctx.get_current_indent() + \
                    f'entry_payload_{i}_i => {entry_payload_i.getNameRead(i)},\n'
        for i in range(0, self.numEntries):
            arch += ctx.get_current_indent() + \
                f'entry_reset_{i}_o => {entry_reset_o.getNameWrite(i)},\n'
        arch += ctx.get_current_indent() + \
            f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
        ctx.tabLevel -= 1
        arch += ctx.get_current_indent() + f');\n'
        ctx.tabLevel -= 1
        return arch
