-----------------------------------------------------------------------------------------
---------------------------------------------------------------- Read support for MemCont
-----------------------------------------------------------------------------------------

---------------------------------------------------------------------
-- read_priority

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity read_priority is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests (pValid signals)
        data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready from next
        priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0)  -- priority function output
    );
end entity;

architecture arch of read_priority is

begin

    process(req, data_ready)
        variable prio_req : std_logic;
    begin
        -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
        priority_out(0) <= req(0) and data_ready(0);
        prio_req        := req(0) and data_ready(0);
        for I in 1 to ARBITER_SIZE - 1 loop
            priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
            prio_req        := prio_req or (req(I) and data_ready(I));
        end loop;
    end process;

end arch;

---------------------------------------------------------------------
-- read_address_mux

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity read_address_mux is
    generic(
        ARBITER_SIZE : natural;
        ADDR_WIDTH   : natural
    );
    port(
        sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        addr_out : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
    );
end entity;

architecture arch of read_address_mux is

begin
    process(sel, addr_in)
        variable addr_out_var : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    begin
        addr_out_var := (others => '0');
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                addr_out_var := addr_in(I);
            end if;
        end loop;
        addr_out <= addr_out_var;
    end process;
end architecture;

---------------------------------------------------------------------
-- read_address_ready

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity read_address_ready is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of read_address_ready is
begin
    GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
        ready(I) <= nReady(I) and sel(I);
    end generate GEN1;
end architecture;

---------------------------------------------------------------------
-- read_data_signals

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity read_data_signals is
    generic(
        ARBITER_SIZE : natural;
        DATA_WIDTH   : natural
    );
    port(
        rst       : in  std_logic;
        clk       : in  std_logic;
        sel       : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        read_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        out_data  : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
        valid     : out std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of read_data_signals is
    signal sel_prev : std_logic_vector(ARBITER_SIZE - 1 downto 0);
    signal out_reg: data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
begin

    process(clk, rst) is
    begin
        if (rst = '1') then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I)    <= '0';
                sel_prev(I) <= '0';
            end loop;
        elsif (rising_edge(clk)) then
            for I in 0 to ARBITER_SIZE - 1 loop
                sel_prev(I) <= sel(I);
                if (sel(I) = '1') then
                    valid(I) <= '1';  --or not nReady(I); -- just sel(I) ??
                    --sel_prev(I) <= '1';
                else
                    if (nReady(I) = '1') then
                        valid(I) <= '0';
                        ---sel_prev(I) <= '0';
                    end if;
                end if;
            end loop;
        end if;
    end process;

    process(clk, rst) is
    begin
        if (rising_edge(clk)) then
         for I in 0 to ARBITER_SIZE - 1 loop
                if (sel_prev(I) = '1') then
                    out_reg(I) <= read_data;
                end if;
         end loop;
         end if;
    end process;

    process(read_data, sel_prev, out_reg) is
    begin
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel_prev(I) = '1') then
                out_data(I) <= read_data;
            else
                out_data(I) <= out_reg(I);
            end if;
        end loop;
    end process;

end architecture;

---------------------------------------------------------------------
-- read_memory_arbiter

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity read_memory_arbiter is
    generic(
        ARBITER_SIZE : natural := 2;
        ADDR_WIDTH   : natural := 32;
        DATA_WIDTH   : natural := 32
    );
    port(
        rst              : in  std_logic;
        clk              : in  std_logic;
        --- interface to previous
        pValid           : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- read requests
        ready            : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready to process read
        address_in       : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        ---interface to next
        nReady           : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can accept data
        valid            : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- sending data to next component
        data_out         : out data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data to next components
        ---interface to memory
        read_enable      : out std_logic;
        read_address     : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
        data_from_memory : in  std_logic_vector(DATA_WIDTH - 1 downto 0));

end entity;

architecture arch of read_memory_arbiter is
    signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

    priority : entity work.read_priority
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            req          => pValid,
            data_ready   => nReady,
            priority_out => priorityOut
        );

    addressing : entity work.read_address_mux
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            ADDR_WIDTH   => ADDR_WIDTH
        )
        port map(
            sel      => priorityOut,
            addr_in  => address_in,
            addr_out => read_address
        );

    addressReady : entity work.read_address_ready
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            sel    => priorityOut,
            nReady => nReady,
            ready  => ready
        );

    data : entity work.read_data_signals
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            DATA_WIDTH   => DATA_WIDTH
        )
        port map(
            rst       => rst,
            clk       => clk,
            sel       => priorityOut,
            read_data => data_from_memory,
            out_data  => data_out,
            valid     => valid,
            nReady    => nReady
        );

    process(priorityOut) is
        variable read_en_var : std_logic;
    begin
        read_en_var := '0';
        for I in 0 to ARBITER_SIZE - 1 loop
            read_en_var := read_en_var or priorityOut(I);
        end loop;
        read_enable <= read_en_var;
    end process;

end architecture;

-----------------------------------------------------------------------------------------
--------------------------------------------------------------- Write support for MemCont
-----------------------------------------------------------------------------------------

---------------------------------------------------------------------
-- write_priority

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity write_priority is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        req          : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        data_ready   : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        priority_out : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );
end entity;

architecture arch of write_priority is

begin

    process(data_ready, req)
        variable prio_req : std_logic;

    begin
        -- the first index I such that (req(I) and data_ready(I) = '1') is '1', others are '0'
        priority_out(0) <= req(0) and data_ready(0);
        prio_req        := req(0) and data_ready(0);

        for I in 1 to ARBITER_SIZE - 1 loop
            priority_out(I) <= (not prio_req) and req(I) and data_ready(I);
            prio_req        := prio_req or (req(I) and data_ready(I));
        end loop;
    end process;
end architecture;

---------------------------------------------------------------------
-- write_address_mux

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity write_address_mux is
    generic(
        ARBITER_SIZE : natural;
        ADDR_WIDTH   : natural
    );
    port(
        sel      : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        addr_in  : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        addr_out : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
    );
end entity;

architecture arch of write_address_mux is

begin
    process(sel, addr_in)
        variable addr_out_var : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    begin
        addr_out_var := (others => '0');
        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                addr_out_var := addr_in(I);
            end if;
        end loop;
        addr_out <= addr_out_var;
    end process;
end architecture;

---------------------------------------------------------------------
-- write_address_ready

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity write_address_ready is
    generic(
        ARBITER_SIZE : natural
    );
    port(
        sel    : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        nReady : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );

end entity;

architecture arch of write_address_ready is

begin

    GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
        ready(I) <= nReady(I) and sel(I);
    end generate GEN1;

end architecture;

---------------------------------------------------------------------
-- write_data_signals

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity write_data_signals is
    generic(
        ARBITER_SIZE : natural;
        DATA_WIDTH   : natural
    );
    port(
        rst        : in  std_logic;
        clk        : in  std_logic;
        sel        : in  std_logic_vector(ARBITER_SIZE - 1 downto 0);
        write_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        in_data    : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
        valid      : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
    );

end entity;

architecture arch of write_data_signals is

begin

    process(sel, in_data)
        variable data_out_var : std_logic_vector(DATA_WIDTH - 1 downto 0);
    begin
        data_out_var := (others => '0');

        for I in 0 to ARBITER_SIZE - 1 loop
            if (sel(I) = '1') then
                data_out_var := in_data(I);
            end if;
        end loop;
        write_data <= data_out_var;
    end process;

    process(clk, rst) is
    begin
        if (rst = '1') then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I) <= '0';
            end loop;

        elsif (rising_edge(clk)) then
            for I in 0 to ARBITER_SIZE - 1 loop
                valid(I) <= sel(I);
            end loop;
        end if;
    end process;
end architecture;

---------------------------------------------------------------------
-- write_memory_arbiter

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity write_memory_arbiter is
    generic(
        ARBITER_SIZE : natural := 2;
        ADDR_WIDTH   : natural := 32;
        DATA_WIDTH   : natural := 32
    );
    port(
        rst            : in  std_logic;
        clk            : in  std_logic;
        --- interface to previous
        pValid         : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); --write requests
        ready          : out std_logic_vector(ARBITER_SIZE - 1 downto 0); -- ready
        address_in     : in  data_array(ARBITER_SIZE - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
        data_in        : in  data_array(ARBITER_SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0); -- data from previous that want to write

        ---interface to next
        nReady         : in  std_logic_vector(ARBITER_SIZE - 1 downto 0); -- next component can continue after write
        valid          : out std_logic_vector(ARBITER_SIZE - 1 downto 0); --sending write confirmation to next component

        ---interface to memory
        write_enable   : out std_logic;
        write_address  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
        data_to_memory : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );

end entity;

architecture arch of write_memory_arbiter is
    signal priorityOut : std_logic_vector(ARBITER_SIZE - 1 downto 0);

begin

    priority : entity work.write_priority
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            req          => pValid,
            data_ready   => nReady,
            priority_out => priorityOut
        );

    addressing : entity work.write_address_mux
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            ADDR_WIDTH   => ADDR_WIDTH
        )
        port map(
            sel      => priorityOut,
            addr_in  => address_in,
            addr_out => write_address
        );

    addressReady : entity work.write_address_ready
        generic map(
            ARBITER_SIZE => ARBITER_SIZE
        )
        port map(
            sel    => priorityOut,
            nReady => nReady,
            ready  => ready
        );
    data : entity work.write_data_signals
        generic map(
            ARBITER_SIZE => ARBITER_SIZE,
            DATA_WIDTH   => DATA_WIDTH
        )
        port map(
            rst        => rst,
            clk        => clk,
            sel        => priorityOut,
            write_data => data_to_memory,
            in_data    => data_in,
            valid      => valid
        );

    process(priorityOut) is
        variable write_en_var : std_logic;
    begin
        write_en_var := '0';
        for I in 0 to ARBITER_SIZE - 1 loop
            write_en_var := write_en_var or priorityOut(I);
        end loop;
        write_enable <= write_en_var;
    end process;
end architecture;

-----------------------------------------------------------------------------------------
--------------------------------------------------------------------------- Load operator
-----------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity mc_load_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;

        -- Input interface
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specInArray   : in  data_array(1 downto 0)(0 downto 0);     -- (addr, data)
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)

        -- Output interface
        output_addr   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specOutArray  : out data_array(1 downto 0)(0 downto 0);     -- (addr, data)
        nReadyArray   : in  std_logic_vector(OUTPUTS - 1 downto 0); -- (addr, data)
        validArray    : out std_logic_vector(OUTPUTS - 1 downto 0)  -- (addr, data)
    );
end mc_load_op;

architecture arch of mc_load_op is

    signal fork_addr_dataIn  : data_array(0 downto 0)(ADDRESS_SIZE - 1 downto 0);
    signal fork_addr_dataOut : data_array(1 downto 0)(ADDRESS_SIZE - 1 downto 0);
    signal fork_addr_ready, fork_addr_pValid : std_logic_vector(0 downto 0);
    signal fork_addr_valid, fork_addr_nReady : std_logic_vector(1 downto 0);

    signal buff_addr_dataIn  : data_array(0 downto 0)(ADDRESS_SIZE - 1 downto 0);
    signal buff_addr_dataOut : data_array(0 downto 0)(ADDRESS_SIZE - 1 downto 0);
    signal buff_addr_ready, buff_addr_pValid : std_logic_vector(0 downto 0);
    signal buff_addr_valid, buff_addr_nReady : std_logic_vector(0 downto 0);

    signal fork_data_dataIn  : data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
    signal fork_data_dataOut : data_array(1 downto 0)(DATA_SIZE - 1 downto 0);
    signal fork_data_ready, fork_data_pValid : std_logic_vector(0 downto 0);
    signal fork_data_valid, fork_data_nReady : std_logic_vector(1 downto 0);

    signal buff_data_dataIn  : data_array(0 downto 0)(DATA_SIZE+1 - 1 downto 0);
    signal buff_data_dataOut : data_array(0 downto 0)(DATA_SIZE+1 - 1 downto 0);
    signal buff_data_ready, buff_data_pValid : std_logic_vector(0 downto 0);
    signal buff_data_valid, buff_data_nReady : std_logic_vector(0 downto 0);

    signal fifo_spec_dataIn  : data_array(0 downto 0)(0 downto 0);
    signal fifo_spec_dataOut : data_array(0 downto 0)(0 downto 0);
    signal fifo_spec_ready, fifo_spec_pValid : std_logic_vector(1 downto 0);
    signal fifo_spec_valid, fifo_spec_nReady : std_logic_vector(0 downto 0);
    --signal fifo_spec_filledSlots : natural range 0 to 1000;
    signal fifo_spec_send : std_logic;

    signal spec_fork_data_dataOut_0 : std_logic_vector(DATA_SIZE+1 - 1 downto 0);

    signal unconnected_spec   : data_array(0 downto 0)(0 downto 0);
    signal unconnected_spec_2 : data_array(1 downto 0)(0 downto 0);

begin

    --------------------------------------------------
    readyArray(1) <= fork_addr_ready(0); -- addr ready
    --------------------------------------------------

    fork_addr_dataIn(0) <= input_addr;
    fork_addr_pValid(0) <= pValidArray(1); -- addr pValid
    fork_addr_nReady <= (fifo_spec_ready(0), buff_addr_ready(0));
    fork_addr: entity work.fork(arch) generic map(1, 2, ADDRESS_SIZE, ADDRESS_SIZE)
        port map (
            clk => clk,
            rst => rst,
            dataInArray         => fork_addr_dataIn,
            specInArray(0)(0)   => '0',
            dataOutArray        => fork_addr_dataOut,
            specOutArray        => unconnected_spec_2,
            pValidArray         => fork_addr_pValid,
            nReadyArray         => fork_addr_nReady,
            validArray          => fork_addr_valid,
            readyArray          => fork_addr_ready
        );

    buff_addr_dataIn(0) <= fork_addr_dataOut(0);
    buff_addr_pValid(0) <= fork_addr_valid(0);
    buff_addr_nReady(0) <= nReadyArray(1);  -- addr nReady
    buff_addr: entity work.TEHB(arch) generic map (1, 1, ADDRESS_SIZE, ADDRESS_SIZE)
        port map (
            clk => clk,
            rst => rst,
            dataInArray         => buff_addr_dataIn,
            specInArray(0)(0)   => '0',
            dataOutArray        => buff_addr_dataOut,
            specOutArray        => unconnected_spec,
            pValidArray         => buff_addr_pValid,
            nReadyArray         => buff_addr_nReady,
            validArray          => buff_addr_valid,
            readyArray          => buff_addr_ready
        );

    --------------------------------------------------
    output_addr   <= buff_addr_dataOut(0);
    validArray(1) <= buff_addr_valid(0); -- addr valid
    --------------------------------------------------

    --------------------------------------------------
    readyArray(0) <= fork_data_ready(0); -- data ready
    --------------------------------------------------

    fork_data_dataIn(0) <= dataInArray(0);
    fork_data_pValid(0) <= pValidArray(0); -- data pValid
    fork_data_nReady <= ('1', buff_data_ready(0)); -- port (1) unused
    fork_data: entity work.fork(arch) generic map(1, 2, DATA_SIZE, DATA_SIZE)
        port map (
            clk => clk,
            rst => rst,
            dataInArray         => fork_data_dataIn,
            specInArray(0)(0)   => '0',
            dataOutArray        => fork_data_dataOut,
            specOutArray        => unconnected_spec_2,
            pValidArray         => fork_data_pValid,
            nReadyArray         => fork_data_nReady,
            validArray          => fork_data_valid,
            readyArray          => fork_data_ready
        );

    ------------------------------------------------------------------------
    spec_fork_data_dataOut_0 <= fifo_spec_dataOut(0) & fork_data_dataOut(0);
    ------------------------------------------------------------------------

    buff_data_dataIn(0) <= spec_fork_data_dataOut_0; --fork_data_dataOut(0);
    buff_data_pValid(0) <= fork_data_valid(0);
    buff_data_nReady(0) <= nReadyArray(0);  -- data nReady
    buff_data: entity work.TEHB(arch) generic map (1, 1, DATA_SIZE+1, DATA_SIZE+1)
        port map (
            clk => clk,
            rst => rst,
            dataInArray         => buff_data_dataIn,
            specInArray(0)(0)   => '0',
            dataOutArray        => buff_data_dataOut,
            specOutArray        => unconnected_spec,
            pValidArray         => buff_data_pValid,
            nReadyArray         => buff_data_nReady,
            validArray          => buff_data_valid,
            readyArray          => buff_data_ready
        );

    ----------------------------------------------------------------
    dataOutArray(0) <= buff_data_dataOut(0)(DATA_SIZE - 1 downto 0); --buff_data_dataOut(0);
    validArray(0)   <= buff_data_valid(0); -- data valid
    ----------------------------------------------------------------

    -- Speculative bit logic -----------------------------------------------------------------
    -- Using a special fifo for spec bits to transfer them from addrIn to dataOut elastically.
    -- Takes in the spec bit when valid addrIn, and doesn't send it out always.
    -- Requires next ready and send = '1' to send the fifo output. Send = dataIn valid.
    -- Hence, fifo sends out spec bit with valid data. It is bundled and sent to buff_data.
    --------------------------------------
    -- specOut for addrOut not relevant
    specOutArray(1)(0) <= '0';
    -- specIn for dataIn always assumed 0
    --------------------------------------
    -- specIn for addrIn generates specOut for dataOut
    --------------------------------------------------

    fifo_spec_send <= fork_data_valid(0) and buff_data_ready(0); -- send when handshake true
    fifo_spec_dataIn(0) <= specInArray(1);  -- specIn for addrIn
    fifo_spec_pValid <= ('1', fork_addr_valid(1));  -- (send, dIn) -- handshake always valid
    fifo_spec_nReady(0) <= '1';  -- assuming ok
    fifo_spec: entity work.load_spec_FIFO(arch) generic map (2, 1, 1, 1, 16)
        port map (
            clk => clk,
            rst => rst,
            dataInArray         => fifo_spec_dataIn,
            dataOutArray        => fifo_spec_dataOut,
            pValidArray         => fifo_spec_pValid,
            nReadyArray         => fifo_spec_nReady,
            validArray          => fifo_spec_valid,
            readyArray          => fifo_spec_ready,  -- (send, dIn)
            --filledSlots         => fifo_spec_filledSlots,
            send                => fifo_spec_send
        );

    ------------------------------------------------------------
    -- specOut for dataOut bundled with data
    specOutArray(0)(0) <= buff_data_dataOut(0)(DATA_SIZE+1 - 1);
    ------------------------------------------------------------

end arch;

-----------------------------------------------------------------------------------------
-------------------------------------------------------------------------- Store operator
-----------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity mc_store_op is
    generic (
        INPUTS        : integer;
        OUTPUTS       : integer;
        ADDRESS_SIZE  : integer;
        DATA_SIZE     : integer
    );
    port (
        clk, rst      : in  std_logic;

        -- Input interface
        input_addr    : in  std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataInArray   : in  data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specInArray   : in  data_array(1 downto 0)(0 downto 0);   -- (addr, data)
        pValidArray   : in  std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)
        readyArray    : out std_logic_vector(INPUTS - 1 downto 0);  -- (addr, data)

        -- Output interface
        output_addr   : out std_logic_vector(ADDRESS_SIZE - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(DATA_SIZE - 1 downto 0);
        specOutArray  : out data_array(1 downto 0)(0 downto 0);   -- (addr, data)
        nReadyArray   : in  std_logic_vector(OUTPUTS - 1 downto 0); -- (addr, data)
        validArray    : out std_logic_vector(OUTPUTS - 1 downto 0)  -- (addr, data)
    );
end mc_store_op;

architecture arch of mc_store_op is

    signal join_valid: std_logic;
    signal spec_out : std_logic;

begin

    -- Join handshake signals
    join_write: entity work.join(arch) generic map (2)
        port map (
            pValidArray => pValidArray,
            nReady      => nReadyArray(0),
            valid       => join_valid,
            readyArray  => readyArray
        );

    -- Send address and data to LSQ
    output_addr <= input_addr;
    validArray(1) <= join_valid;
    ----
    dataOutArray <= dataInArray;
    validArray(0) <= join_valid;

    -- Speculative bit logic
    spec_out <= specInArray(1)(0) or specInArray(0)(0);
    specOutArray(0)(0) <= spec_out;
    specOutArray(1)(0) <= spec_out;

    -- Non-responsive to handshakes -- TODO: check

end arch;


-----------------------------------------------------------------------------------------
------------------------------------------------------------------ Main Memory Controller
-----------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity MemCont is 
    generic ( 
        DATA_SIZE       : natural;
        ADDRESS_SIZE    : natural;
        BB_COUNT        : natural;
        LOAD_COUNT      : natural;
        STORE_COUNT     : natural
    );
    port (
        clk : in  std_logic;
        rst : in  std_logic;
        
        io_storeDataOut         : out std_logic_vector(31 downto 0);
        io_storeAddrOut         : out std_logic_vector(31 downto 0);
        io_storeEnable          : out std_logic;
        io_loadDataIn           : in  std_logic_vector(31 downto 0);
        io_loadAddrOut          : out std_logic_vector(31 downto 0);
        io_loadEnable           : out std_logic;

        io_bb_stCountArray      : in  data_array(BB_COUNT - 1 downto 0)(31 downto 0);
        io_bbpValids            : in  std_logic_vector(BB_COUNT - 1 downto 0);
        io_bbReadyToPrevs       : out std_logic_vector(BB_COUNT - 1 downto 0);

        io_Empty_Valid          : out std_logic;
        io_Empty_Ready          : in  std_logic;

        io_rdPortsPrev_bits     : in  data_array(LOAD_COUNT - 1 downto 0)(ADDRESS_SIZE - 1 downto 0);
        io_rdPortsPrev_valid    : in  std_logic_vector(LOAD_COUNT - 1 downto 0);
        io_rdPortsPrev_ready    : out std_logic_vector(LOAD_COUNT - 1 downto 0);

        io_rdPortsNext_bits     : out data_array(LOAD_COUNT - 1 downto 0)(DATA_SIZE - 1 downto 0);
        io_rdPortsNext_valid    : out std_logic_vector(LOAD_COUNT - 1 downto 0);
        io_rdPortsNext_ready    : in  std_logic_vector(LOAD_COUNT - 1 downto 0);

        io_wrAddrPorts_bits     : in  data_array(STORE_COUNT - 1 downto 0)(ADDRESS_SIZE - 1 downto 0);
        io_wrAddrPorts_valid    : in  std_logic_vector(STORE_COUNT - 1 downto 0);
        io_wrAddrPorts_ready    : out std_logic_vector(STORE_COUNT - 1 downto 0);

        io_wrDataPorts_bits     : in  data_array(STORE_COUNT - 1 downto 0)(DATA_SIZE - 1 downto 0);
        io_wrDataPorts_valid    : in  std_logic_vector(STORE_COUNT - 1 downto 0);
        io_wrDataPorts_ready    : out std_logic_vector(STORE_COUNT - 1 downto 0)
    );

end entity;

architecture arch of MemCont is

    signal counter1 : std_logic_vector(31 downto 0);
    signal valid_WR : std_logic_vector(STORE_COUNT - 1 downto 0);
    constant zero : std_logic_vector(BB_COUNT - 1 downto 0) := (others=>'0');

    signal mcStoreDataOut : std_logic_vector(DATA_SIZE - 1 downto 0);
    signal mcStoreAddrOut : std_logic_vector(ADDRESS_SIZE - 1 downto 0);
    signal mcLoadDataIn   : std_logic_vector(DATA_SIZE - 1 downto 0);
    signal mcLoadAddrOut  : std_logic_vector(ADDRESS_SIZE - 1 downto 0);

begin

    io_wrDataPorts_ready <= io_wrAddrPorts_ready;

    io_storeAddrOut <= std_logic_vector (resize(unsigned(mcStoreAddrOut),io_storeDataOut'length));
    io_storeDataOut <= std_logic_vector (resize(unsigned(mcStoreDataOut),io_storeDataOut'length));
    io_loadAddrOut  <= std_logic_vector (resize(unsigned(mcLoadAddrOut),io_loadAddrOut'length));
    mcLoadDataIn    <= std_logic_vector (resize(unsigned(io_loadDataIn),mcLoadDataIn'length));

    read_arbiter : entity work.read_memory_arbiter
        generic map (
            ARBITER_SIZE => LOAD_COUNT,
            ADDR_WIDTH   => ADDRESS_SIZE,
            DATA_WIDTH   => DATA_SIZE
        )
        port map (
            rst              => rst,
            clk              => clk,
            pValid           => io_rdPortsPrev_valid,
            ready            => io_rdPortsPrev_ready,
            address_in       => io_rdPortsPrev_bits, -- if two address lines are presented change this to corresponding one.
            nReady           => io_rdPortsNext_ready,
            valid            => io_rdPortsNext_valid,
            data_out         => io_rdPortsNext_bits,
            read_enable      => io_loadEnable,
            read_address     => mcLoadAddrOut,
            data_from_memory => mcLoadDataIn
        );

    write_arbiter : entity work.write_memory_arbiter
        generic map (
            ARBITER_SIZE => STORE_COUNT,
            ADDR_WIDTH   => ADDRESS_SIZE,
            DATA_WIDTH   => DATA_SIZE
        )
        port map (
            rst            => rst,
            clk            => clk,
            pValid         => io_wrAddrPorts_valid,
            ready          => io_wrAddrPorts_ready,
            address_in     => io_wrAddrPorts_bits, -- if two address lines are presented change this to corresponding one.
            data_in        => io_wrDataPorts_bits,
            nReady         => (others => '1'), --for now, setting as always ready
            valid          => valid_WR, -- unconnected
            write_enable   => io_storeEnable,
            write_address  => mcStoreAddrOut,
            data_to_memory => mcStoreDataOut
        );

    Counter: process (clk)
        variable counter : std_logic_vector(31 downto 0);
    begin
        if (rst = '1') then
            counter := (31 downto 0 => '0');
          
        elsif rising_edge(clk) then
            -- increment counter by number of stores in BB
            for I in 0 to BB_COUNT - 1 loop
                if (io_bbpValids(I) = '1') then
                    counter := std_logic_vector(unsigned(counter) + unsigned(io_bb_stCountArray(I)));
                end if;
            end loop;

            -- decrement counter whenever store issued to memory
            if (io_StoreEnable = '1') then
                counter := std_logic_vector(unsigned(counter) - 1);
            end if;

            counter1 <= counter;
        end if;
    end process;

    -- Check if there are any outstanding store requests. If not, program can terminate.
    io_Empty_Valid <= '1' when (counter1 = (31 downto 0 => '0') and (io_bbpValids(BB_COUNT - 1 downto 0) = zero)) else '0';

    -- Always ready to increment counter
    io_bbReadyToPrevs <= (others => '1');

end architecture;
