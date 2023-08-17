library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;
entity mem_controller is generic( DATA_BITWIDTH: natural; ADDR_BITWIDTH: natural; LOAD_COUNT : natural; STORE_COUNT: natural);
port (
    rst: in std_logic;
    clk: in std_logic;
    io_storeDataOut    : out std_logic_vector(31 downto 0);
    io_storeAddrOut    : out std_logic_vector(31 downto 0);
    io_storeEnable : out std_logic;
    io_loadDataIn : in std_logic_vector(31 downto 0);
    io_loadAddrOut : out std_logic_vector(31 downto 0);
    io_loadEnable : out std_logic;

    io_bbpValids : in std_logic_vector(0 downto 0);
    io_bb_stCountArray: in data_array (0 downto 0)(31 downto 0);
    io_bbReadyToPrevs : out std_logic_vector(0 downto 0);

    io_Empty_Valid: out STD_LOGIC;
    io_Empty_Ready: in STD_LOGIC;

    io_rdPortsPrev_valid : in  std_logic_vector(LOAD_COUNT - 1 downto 0);
    io_rdPortsPrev_bits: in data_array (LOAD_COUNT - 1 downto 0)(ADDR_BITWIDTH -1 downto 0);
    io_rdPortsPrev_ready : out  std_logic_vector(LOAD_COUNT - 1 downto 0);

    io_rdPortsNext_bits: out data_array (LOAD_COUNT - 1 downto 0)(DATA_BITWIDTH -1 downto 0);
    io_rdPortsNext_valid : out  std_logic_vector(LOAD_COUNT - 1 downto 0);
    io_rdPortsNext_ready: in  std_logic_vector(LOAD_COUNT - 1 downto 0);

    io_wrAddrPorts_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    io_wrAddrPorts_bits: in data_array (STORE_COUNT - 1 downto 0)(ADDR_BITWIDTH -1 downto 0);
    io_wrAddrPorts_ready : out  std_logic_vector(STORE_COUNT - 1 downto 0);

    io_wrDataPorts_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    io_wrDataPorts_bits: in data_array (STORE_COUNT - 1 downto 0)(DATA_BITWIDTH -1 downto 0);
    io_wrDataPorts_ready : out  std_logic_vector(STORE_COUNT - 1 downto 0)
    );

end entity;


architecture arch of mem_controller is
signal counter1 : std_logic_vector(31 downto 0);
signal valid_WR: std_logic_vector(STORE_COUNT - 1 downto 0);
constant zero: std_logic_vector(0 downto 0) := (others=>'0');

signal mcStoreDataOut : std_logic_vector(DATA_BITWIDTH-1 downto 0);
signal mcStoreAddrOut :  std_logic_vector(ADDR_BITWIDTH-1 downto 0);
signal mcLoadDataIn : std_logic_vector(DATA_BITWIDTH-1 downto 0);
signal mcLoadAddrOut : std_logic_vector(ADDR_BITWIDTH-1 downto 0);

begin
 io_wrDataPorts_ready<= io_wrAddrPorts_ready;



 io_storeDataOut <= std_logic_vector (resize(unsigned(mcStoreDataOut),io_storeDataOut'length));
 io_storeAddrOut <= std_logic_vector (resize(unsigned(mcStoreAddrOut),io_storeDataOut'length));
 mcLoadDataIn <= std_logic_vector (resize(unsigned(io_loadDataIn),mcLoadDataIn'length));
 io_loadAddrOut <= std_logic_vector (resize(unsigned(mcLoadAddrOut),io_loadAddrOut'length));

    read_arbiter : entity work.read_memory_arbiter
        generic map(
            ARBITER_SIZE => LOAD_COUNT,
            ADDR_WIDTH   => ADDR_BITWIDTH,
            DATA_WIDTH   => DATA_BITWIDTH
        )
        port map(
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
            data_from_memory =>  mcLoadDataIn
        );

    write_arbiter : entity work.write_memory_arbiter
        generic map(
            ARBITER_SIZE => STORE_COUNT,
            ADDR_WIDTH   => ADDR_BITWIDTH,
            DATA_WIDTH   => DATA_BITWIDTH
        )
        port map(
            rst            => rst,
            clk            => clk,
            pValid         => io_wrAddrPorts_valid,
            ready          => io_wrAddrPorts_ready,
            address_in     => io_wrAddrPorts_bits, -- if two address lines are presented change this to corresponding one.
            data_in        => io_wrDataPorts_bits,
            nReady         => (others => '1'), --for now, setting as always ready
            valid          => valid_WR, -- unconnected
            write_enable   => io_storeEnable,
            --enable         => io_storeEnable,
            write_address  => mcStoreAddrOut,
            data_to_memory => mcStoreDataOut
        );

        Counter: process (CLK)
        variable counter : std_logic_vector(31 downto 0);
        begin
            if (rst = '1') then
                counter:=(31 downto 0 => '0');
              
            elsif rising_edge(CLK) then
                -- increment counter by number of stores in BB
                for I in 0 to 0 loop
                    if (io_bbpValids(I) = '1') then
                        counter:= std_logic_vector(unsigned(counter) + unsigned(io_bb_stCountArray(I)));
                    end if;
                end loop;

                -- decrement counter whenever store issued to memory
                if (io_StoreEnable = '1') then
                    counter:= std_logic_vector(unsigned(counter) - 1);
                end if;

                counter1 <= counter;
            end if;

        end process;

        -- check if there are any outstanding store requests
        -- if not, program can terminate
        io_Empty_Valid <= '1' when (counter1 =  (31 downto 0 => '0') and (io_bbpValids(0 downto 0)  = zero))  else '0'; 

        io_bbReadyToPrevs <= (others => '1'); -- always ready to increment counter;

end architecture;
