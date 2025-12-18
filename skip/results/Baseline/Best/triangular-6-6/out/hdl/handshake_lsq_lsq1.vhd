

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1 is
	port(
		reset : in std_logic;
		clock : in std_logic;
		io_stDataToMC_bits : out std_logic_vector(31 downto 0);
		io_stAddrToMC_bits : out std_logic_vector(6 downto 0);
		io_ldDataFromMC_bits : in std_logic_vector(31 downto 0);
		io_ldAddrToMC_bits : out std_logic_vector(6 downto 0);
		io_ctrl_0_ready : out std_logic;
		io_ctrl_0_valid : in std_logic;
		io_ldAddr_0_ready : out std_logic;
		io_ldAddr_0_valid : in std_logic;
		io_ldAddr_0_bits : in std_logic_vector(6 downto 0);
		io_ldData_0_ready : in std_logic;
		io_ldData_0_valid : out std_logic;
		io_ldData_0_bits : out std_logic_vector(31 downto 0);
		io_stAddr_0_ready : out std_logic;
		io_stAddr_0_valid : in std_logic;
		io_stAddr_0_bits : in std_logic_vector(6 downto 0);
		io_stData_0_ready : out std_logic;
		io_stData_0_valid : in std_logic;
		io_stData_0_bits : in std_logic_vector(31 downto 0);
		io_ldAddrToMC_ready : in std_logic;
		io_ldAddrToMC_valid : out std_logic;
		io_ldDataFromMC_ready : out std_logic;
		io_ldDataFromMC_valid : in std_logic;
		io_stAddrToMC_ready : in std_logic;
		io_stAddrToMC_valid : out std_logic;
		io_stDataToMC_ready : in std_logic;
		io_stDataToMC_valid : out std_logic
	);
end entity;

architecture arch of handshake_lsq_lsq1 is
	signal io_loadEn : std_logic;
	signal io_storeEn : std_logic;
	signal rresp_0_id : std_logic_vector(3 downto 0);
	signal wreq_0_ready : std_logic;
	signal wresp_0_valid : std_logic;
	signal wresp_0_id : std_logic_vector(3 downto 0);
	signal rreq_0_id : std_logic_vector(3 downto 0);
	signal wreq_0_id : std_logic_vector(3 downto 0);
begin
	----------------------------------------------------------------------------
	-- Process for rresp_id
	process (clock, reset) is
	begin
		if reset = '1' then
			rresp_0_id <= ( others => '0' );
		elsif rising_edge(clock) then

			if io_loadEn = '1' then
				rresp_0_id <= rreq_0_id;
			end if;
		end if;
	end process;
	----------------------------------------------------------------------------
	----------------------------------------------------------------------------
	-- Process for wreq_ready, wresp_valid and wresp_id
	process (clock, reset) is
	begin
		if reset = '1' then
			wresp_0_valid <= '0';
			wresp_0_id <= ( others => '0' );
		elsif rising_edge(clock) then

			if io_storeEn = '1' then
				wresp_0_valid <= '1';
				wresp_0_id <= rreq_0_id;
			else
				wresp_0_valid <= '0';
			end if;
		end if;
	end process;
	----------------------------------------------------------------------------
	-- Signal Assignment
	io_ldAddrToMC_valid <= io_loadEn;
	io_stAddrToMC_valid <= io_storeEn;
	io_stDataToMC_valid <= io_storeEn;
	wreq_0_ready <= io_stAddrToMC_ready and io_stDataToMC_ready;
	-- Instantiate the core LSQ logic
	handshake_lsq_lsq1_core : entity work.handshake_lsq_lsq1_core
		port map(
			rst => reset,
			clk => clock,
			wreq_data_0_o => io_stDataToMC_bits,
			wreq_addr_0_o => io_stAddrToMC_bits,
			wreq_valid_0_o => io_storeEn,
			rresp_data_0_i => io_ldDataFromMC_bits,
			rreq_addr_0_o => io_ldAddrToMC_bits,
			rreq_valid_0_o => io_loadEn,
			group_init_ready_0_o => io_ctrl_0_ready,
			group_init_valid_0_i => io_ctrl_0_valid,
			ldp_addr_ready_0_o => io_ldAddr_0_ready,
			ldp_addr_valid_0_i => io_ldAddr_0_valid,
			ldp_addr_0_i => io_ldAddr_0_bits,
			ldp_data_ready_0_i => io_ldData_0_ready,
			ldp_data_valid_0_o => io_ldData_0_valid,
			ldp_data_0_o => io_ldData_0_bits,
			stp_addr_ready_0_o => io_stAddr_0_ready,
			stp_addr_valid_0_i => io_stAddr_0_valid,
			stp_addr_0_i => io_stAddr_0_bits,
			stp_data_ready_0_o => io_stData_0_ready,
			stp_data_valid_0_i => io_stData_0_valid,
			stp_data_0_i => io_stData_0_bits,
			rreq_ready_0_i => io_ldAddrToMC_ready,
			rresp_valid_0_i => io_ldDataFromMC_valid,
			rresp_ready_0_o => io_ldDataFromMC_ready,
			rresp_id_0_i => rresp_0_id,
			rreq_id_0_o => rreq_0_id,
			wreq_ready_0_i => wreq_0_ready,
			wresp_valid_0_i => wresp_0_valid,
			wresp_id_0_i => wresp_0_id,
			wreq_id_0_o => wreq_0_id
		);
end architecture;
