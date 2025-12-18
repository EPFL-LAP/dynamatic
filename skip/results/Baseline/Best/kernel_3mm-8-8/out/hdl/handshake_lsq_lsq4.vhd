

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq4 is
	port(
		reset : in std_logic;
		clock : in std_logic;
		io_storeData : out std_logic_vector(31 downto 0);
		io_storeAddr : out std_logic_vector(6 downto 0);
		io_storeEn : out std_logic;
		io_loadData : in std_logic_vector(31 downto 0);
		io_loadAddr : out std_logic_vector(6 downto 0);
		io_loadEn : out std_logic;
		io_ctrl_0_ready : out std_logic;
		io_ctrl_1_ready : out std_logic;
		io_ctrl_2_ready : out std_logic;
		io_ctrl_0_valid : in std_logic;
		io_ctrl_1_valid : in std_logic;
		io_ctrl_2_valid : in std_logic;
		io_ldAddr_0_ready : out std_logic;
		io_ldAddr_1_ready : out std_logic;
		io_ldAddr_0_valid : in std_logic;
		io_ldAddr_1_valid : in std_logic;
		io_ldAddr_0_bits : in std_logic_vector(6 downto 0);
		io_ldAddr_1_bits : in std_logic_vector(6 downto 0);
		io_ldData_0_ready : in std_logic;
		io_ldData_1_ready : in std_logic;
		io_ldData_0_valid : out std_logic;
		io_ldData_1_valid : out std_logic;
		io_ldData_0_bits : out std_logic_vector(31 downto 0);
		io_ldData_1_bits : out std_logic_vector(31 downto 0);
		io_stAddr_0_ready : out std_logic;
		io_stAddr_1_ready : out std_logic;
		io_stAddr_0_valid : in std_logic;
		io_stAddr_1_valid : in std_logic;
		io_stAddr_0_bits : in std_logic_vector(6 downto 0);
		io_stAddr_1_bits : in std_logic_vector(6 downto 0);
		io_stData_0_ready : out std_logic;
		io_stData_1_ready : out std_logic;
		io_stData_0_valid : in std_logic;
		io_stData_1_valid : in std_logic;
		io_stData_0_bits : in std_logic_vector(31 downto 0);
		io_stData_1_bits : in std_logic_vector(31 downto 0);
		io_memStart_ready : out std_logic;
		io_memStart_valid : in std_logic;
		io_ctrlEnd_ready : out std_logic;
		io_ctrlEnd_valid : in std_logic;
		io_memEnd_ready : in std_logic;
		io_memEnd_valid : out std_logic
	);
end entity;

architecture arch of handshake_lsq_lsq4 is
	signal rreq_0_ready : std_logic;
	signal rresp_0_valid : std_logic;
	signal rresp_0_id : std_logic_vector(3 downto 0);
	signal wreq_0_ready : std_logic;
	signal wresp_0_valid : std_logic;
	signal wresp_0_id : std_logic_vector(3 downto 0);
	signal rreq_0_id : std_logic_vector(3 downto 0);
	signal wreq_0_id : std_logic_vector(3 downto 0);
begin
	----------------------------------------------------------------------------
	-- Process for rreq_ready, rresp_valid and rresp_id
	process (clock, reset) is
	begin
		if reset = '1' then
			rreq_0_ready <= '0';
			rresp_0_valid <= '0';
			rresp_0_id <= ( others => '0' );
		elsif rising_edge(clock) then
			rreq_0_ready <= '1';

			if io_loadEn = '1' then
				rresp_0_valid <= '1';
				rresp_0_id <= rreq_0_id;
			else
				rresp_0_valid <= '0';
			end if;
		end if;
	end process;
	----------------------------------------------------------------------------
	----------------------------------------------------------------------------
	-- Process for wreq_ready, wresp_valid and wresp_id
	process (clock, reset) is
	begin
		if reset = '1' then
			wreq_0_ready <= '0';
			wresp_0_valid <= '0';
			wresp_0_id <= ( others => '0' );
		elsif rising_edge(clock) then
			wreq_0_ready <= '1';

			if io_storeEn = '1' then
				wresp_0_valid <= '1';
				wresp_0_id <= rreq_0_id;
			else
				wresp_0_valid <= '0';
			end if;
		end if;
	end process;
	----------------------------------------------------------------------------
	-- Instantiate the core LSQ logic
	handshake_lsq_lsq4_core : entity work.handshake_lsq_lsq4_core
		port map(
			rst => reset,
			clk => clock,
			wreq_data_0_o => io_storeData,
			wreq_addr_0_o => io_storeAddr,
			wreq_valid_0_o => io_storeEn,
			rresp_data_0_i => io_loadData,
			rreq_addr_0_o => io_loadAddr,
			rreq_valid_0_o => io_loadEn,
			memStart_ready_o => io_memStart_ready,
			memStart_valid_i => io_memStart_valid,
			ctrlEnd_ready_o => io_ctrlEnd_ready,
			ctrlEnd_valid_i => io_ctrlEnd_valid,
			memEnd_ready_i => io_memEnd_ready,
			memEnd_valid_o => io_memEnd_valid,
			group_init_ready_0_o => io_ctrl_0_ready,
			group_init_valid_0_i => io_ctrl_0_valid,
			group_init_ready_1_o => io_ctrl_1_ready,
			group_init_valid_1_i => io_ctrl_1_valid,
			group_init_ready_2_o => io_ctrl_2_ready,
			group_init_valid_2_i => io_ctrl_2_valid,
			ldp_addr_ready_0_o => io_ldAddr_0_ready,
			ldp_addr_valid_0_i => io_ldAddr_0_valid,
			ldp_addr_0_i => io_ldAddr_0_bits,
			ldp_data_ready_0_i => io_ldData_0_ready,
			ldp_data_valid_0_o => io_ldData_0_valid,
			ldp_data_0_o => io_ldData_0_bits,
			ldp_addr_ready_1_o => io_ldAddr_1_ready,
			ldp_addr_valid_1_i => io_ldAddr_1_valid,
			ldp_addr_1_i => io_ldAddr_1_bits,
			ldp_data_ready_1_i => io_ldData_1_ready,
			ldp_data_valid_1_o => io_ldData_1_valid,
			ldp_data_1_o => io_ldData_1_bits,
			stp_addr_ready_0_o => io_stAddr_0_ready,
			stp_addr_valid_0_i => io_stAddr_0_valid,
			stp_addr_0_i => io_stAddr_0_bits,
			stp_data_ready_0_o => io_stData_0_ready,
			stp_data_valid_0_i => io_stData_0_valid,
			stp_data_0_i => io_stData_0_bits,
			stp_addr_ready_1_o => io_stAddr_1_ready,
			stp_addr_valid_1_i => io_stAddr_1_valid,
			stp_addr_1_i => io_stAddr_1_bits,
			stp_data_ready_1_o => io_stData_1_ready,
			stp_data_valid_1_i => io_stData_1_valid,
			stp_data_1_i => io_stData_1_bits,
			rreq_ready_0_i => rreq_0_ready,
			rresp_valid_0_i => rresp_0_valid,
			rresp_id_0_i => rresp_0_id,
			rreq_id_0_o => rreq_0_id,
			wreq_ready_0_i => wreq_0_ready,
			wresp_valid_0_i => wresp_0_valid,
			wresp_id_0_i => wresp_0_id,
			wreq_id_0_o => wreq_0_id
		);
end architecture;
