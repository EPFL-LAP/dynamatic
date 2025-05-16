

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldq_tail_i : in std_logic_vector(0 downto 0);
		ldq_head_i : in std_logic_vector(0 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(0 downto 0);
		stq_head_i : in std_logic_vector(0 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		num_loads_o : out std_logic_vector(1 downto 0);
		ldq_port_idx_0_o : out std_logic_vector(0 downto 0);
		ldq_port_idx_1_o : out std_logic_vector(0 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		num_stores_o : out std_logic_vector(1 downto 0);
		stq_port_idx_0_o : out std_logic_vector(0 downto 0);
		stq_port_idx_1_o : out std_logic_vector(0 downto 0);
		ga_ls_order_0_o : out std_logic_vector(1 downto 0);
		ga_ls_order_1_o : out std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of config_0_core_ga is
	signal num_loads : std_logic_vector(1 downto 0);
	signal num_stores : std_logic_vector(1 downto 0);
	signal loads_sub : std_logic_vector(0 downto 0);
	signal stores_sub : std_logic_vector(0 downto 0);
	signal empty_loads : std_logic_vector(1 downto 0);
	signal empty_stores : std_logic_vector(1 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal ldq_port_idx_rom_0 : std_logic_vector(0 downto 0);
	signal ldq_port_idx_rom_1 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_0 : std_logic_vector(0 downto 0);
	signal stq_port_idx_rom_1 : std_logic_vector(0 downto 0);
	signal ga_ls_order_rom_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_1_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_0_0 : std_logic_vector(0 downto 0);
	signal TEMP_2_mux_1_0 : std_logic_vector(0 downto 0);
	signal TEMP_3_mux_0_0 : std_logic_vector(1 downto 0);
	signal TEMP_3_mux_1_0 : std_logic_vector(1 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(1 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(1 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 2)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 2)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i));
	-- WrapAdd End

	empty_loads <= "10" when ldq_empty_i else ( '0' & loads_sub );
	empty_stores <= "10" when stq_empty_i else ( '0' & stores_sub );
	group_init_ready_0 <= '1' when ( empty_loads >= "10" ) and ( empty_stores >= "10" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ldq_port_idx_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "0";
	ldq_port_idx_rom_0 <= TEMP_1_mux_0_0;
	-- Loop 1
	TEMP_1_mux_1_0 <= "1" when group_init_hs_0 else "0";
	ldq_port_idx_rom_1 <= TEMP_1_mux_1_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(stq_port_idx_rom, group_init_hs)
	-- Loop 0
	TEMP_2_mux_0_0 <= "0";
	stq_port_idx_rom_0 <= TEMP_2_mux_0_0;
	-- Loop 1
	TEMP_2_mux_1_0 <= "1" when group_init_hs_0 else "0";
	stq_port_idx_rom_1 <= TEMP_2_mux_1_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_3_mux_0_0 <= "01" when group_init_hs_0 else "00";
	ga_ls_order_rom_0 <= TEMP_3_mux_0_0;
	-- Loop 1
	TEMP_3_mux_1_0 <= "11" when group_init_hs_0 else "00";
	ga_ls_order_rom_1 <= TEMP_3_mux_1_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_4_mux_0 <= "10" when group_init_hs_0 else "00";
	num_loads <= TEMP_4_mux_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_5_mux_0 <= "10" when group_init_hs_0 else "00";
	num_stores <= TEMP_5_mux_0;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "0" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "1" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "0" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "1" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_port_idx, ldq_port_idx_rom, ldq_tail)
	ldq_port_idx_0_o <= ldq_port_idx_rom_1 when ldq_tail_i(0) else ldq_port_idx_rom_0;
	ldq_port_idx_1_o <= ldq_port_idx_rom_0 when ldq_tail_i(0) else ldq_port_idx_rom_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_port_idx, stq_port_idx_rom, stq_tail)
	stq_port_idx_0_o <= stq_port_idx_rom_1 when stq_tail_i(0) else stq_port_idx_rom_0;
	stq_port_idx_1_o <= stq_port_idx_rom_0 when stq_tail_i(0) else stq_port_idx_rom_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	ldq_wen_0_o <= ldq_wen_unshifted_1 when ldq_tail_i(0) else ldq_wen_unshifted_0;
	ldq_wen_1_o <= ldq_wen_unshifted_0 when ldq_tail_i(0) else ldq_wen_unshifted_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	stq_wen_0_o <= stq_wen_unshifted_1 when stq_tail_i(0) else stq_wen_unshifted_0;
	stq_wen_1_o <= stq_wen_unshifted_0 when stq_tail_i(0) else stq_wen_unshifted_1;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	ga_ls_order_temp_0(0) <= ga_ls_order_rom_0(1) when stq_tail_i(0) else ga_ls_order_rom_0(0);
	ga_ls_order_temp_0(1) <= ga_ls_order_rom_0(0) when stq_tail_i(0) else ga_ls_order_rom_0(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	ga_ls_order_temp_1(0) <= ga_ls_order_rom_1(1) when stq_tail_i(0) else ga_ls_order_rom_1(0);
	ga_ls_order_temp_1(1) <= ga_ls_order_rom_1(0) when stq_tail_i(0) else ga_ls_order_rom_1(1);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	ga_ls_order_0_o <= ga_ls_order_temp_1 when ldq_tail_i(0) else ga_ls_order_temp_0;
	ga_ls_order_1_o <= ga_ls_order_temp_0 when ldq_tail_i(0) else ga_ls_order_temp_1;
	-- Shifter End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core_lda is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_bits_0_i : in std_logic_vector(9 downto 0);
		port_bits_1_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_valid_0_i : in std_logic;
		entry_valid_1_i : in std_logic;
		entry_bits_valid_0_i : in std_logic;
		entry_bits_valid_1_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_bits_0_o : out std_logic_vector(9 downto 0);
		entry_bits_1_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of config_0_core_lda is
	signal entry_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_port_valid_1 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(9 downto 0);
	signal entry_request_valid_0 : std_logic;
	signal entry_request_valid_1 : std_logic;
	signal entry_port_request_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_1 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal entry_port_and_0 : std_logic_vector(1 downto 0);
	signal entry_port_and_1 : std_logic_vector(1 downto 0);
	signal entry_port_hs_0 : std_logic_vector(1 downto 0);
	signal entry_port_hs_1 : std_logic_vector(1 downto 0);
	signal TEMP_3_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_in_1 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_1 : std_logic_vector(3 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_0, entry_port_idx_0)
	entry_port_valid_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_valid_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_1, entry_port_idx_1)
	entry_port_valid_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_valid_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_bits_0, port_bits, entry_port_valid_0)
	TEMP_1_mux_0 <= port_bits_0_i when entry_port_valid_0(0) = '1' else "0000000000";
	TEMP_1_mux_1 <= port_bits_1_i when entry_port_valid_0(1) = '1' else "0000000000";
	entry_bits_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_bits_1, port_bits, entry_port_valid_1)
	TEMP_2_mux_0 <= port_bits_0_i when entry_port_valid_1(0) = '1' else "0000000000";
	TEMP_2_mux_1 <= port_bits_1_i when entry_port_valid_1(1) = '1' else "0000000000";
	entry_bits_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	entry_request_valid_0 <= entry_valid_0_i and not entry_bits_valid_0_i;
	entry_request_valid_1 <= entry_valid_1_i and not entry_bits_valid_1_i;
	entry_port_request_0 <= entry_port_valid_0 when entry_request_valid_0 else "00";
	entry_port_request_1 <= entry_port_valid_1 when entry_request_valid_1 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_port_request, or)
	port_ready_vec <= entry_port_request_0 or entry_port_request_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	port_ready_1_o <= port_ready_vec(1);
	entry_port_and_0(0) <= entry_port_request_0(0) and port_valid_0_i;
	entry_port_and_0(1) <= entry_port_request_0(1) and port_valid_1_i;
	entry_port_and_1(0) <= entry_port_request_1(0) and port_valid_0_i;
	entry_port_and_1(1) <= entry_port_request_1(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_hs, entry_port_and, queue_head_oh)
	TEMP_3_double_in_0(0) <= entry_port_and_0(0);
	TEMP_3_double_in_0(2) <= entry_port_and_0(0);
	TEMP_3_double_in_0(1) <= entry_port_and_1(0);
	TEMP_3_double_in_0(3) <= entry_port_and_1(0);
	TEMP_3_double_out_0 <= TEMP_3_double_in_0 and not std_logic_vector( unsigned( TEMP_3_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(0) <= TEMP_3_double_out_0(0) or TEMP_3_double_out_0(2);
	entry_port_hs_1(0) <= TEMP_3_double_out_0(1) or TEMP_3_double_out_0(3);
	TEMP_3_double_in_1(0) <= entry_port_and_0(1);
	TEMP_3_double_in_1(2) <= entry_port_and_0(1);
	TEMP_3_double_in_1(1) <= entry_port_and_1(1);
	TEMP_3_double_in_1(3) <= entry_port_and_1(1);
	TEMP_3_double_out_1 <= TEMP_3_double_in_1 and not std_logic_vector( unsigned( TEMP_3_double_in_1 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(1) <= TEMP_3_double_out_1(0) or TEMP_3_double_out_1(2);
	entry_port_hs_1(1) <= TEMP_3_double_out_1(1) or TEMP_3_double_out_1(3);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_hs_0, or)
	entry_wen_0_o <= entry_port_hs_0(0) or entry_port_hs_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_hs_1, or)
	entry_wen_1_o <= entry_port_hs_1(0) or entry_port_hs_1(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core_ldd is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_bits_0_o : out std_logic_vector(12 downto 0);
		port_bits_1_o : out std_logic_vector(12 downto 0);
		port_valid_0_o : out std_logic;
		port_valid_1_o : out std_logic;
		port_ready_0_i : in std_logic;
		port_ready_1_i : in std_logic;
		entry_valid_0_i : in std_logic;
		entry_valid_1_i : in std_logic;
		entry_bits_valid_0_i : in std_logic;
		entry_bits_valid_1_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_bits_0_i : in std_logic_vector(12 downto 0);
		entry_bits_1_i : in std_logic_vector(12 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of config_0_core_ldd is
	signal entry_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_port_valid_1 : std_logic_vector(1 downto 0);
	signal entry_port_request_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_1 : std_logic_vector(1 downto 0);
	signal entry_port_request_prio_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_prio_1 : std_logic_vector(1 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(3 downto 0);
	signal TEMP_1_double_in_1 : std_logic_vector(3 downto 0);
	signal TEMP_1_double_out_1 : std_logic_vector(3 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(12 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_3_mux_1 : std_logic_vector(12 downto 0);
	signal entry_port_request_valid_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_valid_1 : std_logic_vector(1 downto 0);
	signal port_valid_vec : std_logic_vector(1 downto 0);
	signal entry_port_hs_0 : std_logic_vector(1 downto 0);
	signal entry_port_hs_1 : std_logic_vector(1 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_0, entry_port_idx_0)
	entry_port_valid_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_valid_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_1, entry_port_idx_1)
	entry_port_valid_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_valid_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	entry_port_request_0 <= entry_port_valid_0 when entry_valid_0_i else "00";
	entry_port_request_1 <= entry_port_valid_1 when entry_valid_1_i else "00";
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_request_prio, entry_port_request, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_port_request_0(0);
	TEMP_1_double_in_0(2) <= entry_port_request_0(0);
	TEMP_1_double_in_0(1) <= entry_port_request_1(0);
	TEMP_1_double_in_0(3) <= entry_port_request_1(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_request_prio_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(2);
	entry_port_request_prio_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(3);
	TEMP_1_double_in_1(0) <= entry_port_request_0(1);
	TEMP_1_double_in_1(2) <= entry_port_request_0(1);
	TEMP_1_double_in_1(1) <= entry_port_request_1(1);
	TEMP_1_double_in_1(3) <= entry_port_request_1(1);
	TEMP_1_double_out_1 <= TEMP_1_double_in_1 and not std_logic_vector( unsigned( TEMP_1_double_in_1 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_request_prio_0(1) <= TEMP_1_double_out_1(0) or TEMP_1_double_out_1(2);
	entry_port_request_prio_1(1) <= TEMP_1_double_out_1(1) or TEMP_1_double_out_1(3);
	-- Priority Masking End

	-- Mux1H Begin
	-- Mux1H(port_bits_0, entry_bits, entry_port_request_prio)
	TEMP_2_mux_0 <= entry_bits_0_i when entry_port_request_prio_0(0) = '1' else "0000000000000";
	TEMP_2_mux_1 <= entry_bits_1_i when entry_port_request_prio_1(0) = '1' else "0000000000000";
	port_bits_0_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(port_bits_1, entry_bits, entry_port_request_prio)
	TEMP_3_mux_0 <= entry_bits_0_i when entry_port_request_prio_0(1) = '1' else "0000000000000";
	TEMP_3_mux_1 <= entry_bits_1_i when entry_port_request_prio_1(1) = '1' else "0000000000000";
	port_bits_1_o <= TEMP_3_mux_0 or TEMP_3_mux_1;
	-- Mux1H End

	entry_port_request_valid_0 <= entry_port_request_prio_0 when entry_bits_valid_0_i else "00";
	entry_port_request_valid_1 <= entry_port_request_prio_1 when entry_bits_valid_1_i else "00";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_port_request_valid, or)
	port_valid_vec <= entry_port_request_valid_0 or entry_port_request_valid_1;
	-- Reduction End

	port_valid_0_o <= port_valid_vec(0);
	port_valid_1_o <= port_valid_vec(1);
	entry_port_hs_0(0) <= entry_port_request_valid_0(0) and port_ready_0_i;
	entry_port_hs_0(1) <= entry_port_request_valid_0(1) and port_ready_1_i;
	entry_port_hs_1(0) <= entry_port_request_valid_1(0) and port_ready_0_i;
	entry_port_hs_1(1) <= entry_port_request_valid_1(1) and port_ready_1_i;
	-- Reduction Begin
	-- Reduce(entry_reset_0, entry_port_hs_0, or)
	entry_reset_0_o <= entry_port_hs_0(0) or entry_port_hs_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_1, entry_port_hs_1, or)
	entry_reset_1_o <= entry_port_hs_1(0) or entry_port_hs_1(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core_sta is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_bits_0_i : in std_logic_vector(9 downto 0);
		port_bits_1_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_valid_0_i : in std_logic;
		entry_valid_1_i : in std_logic;
		entry_bits_valid_0_i : in std_logic;
		entry_bits_valid_1_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_bits_0_o : out std_logic_vector(9 downto 0);
		entry_bits_1_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of config_0_core_sta is
	signal entry_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_port_valid_1 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(9 downto 0);
	signal entry_request_valid_0 : std_logic;
	signal entry_request_valid_1 : std_logic;
	signal entry_port_request_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_1 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal entry_port_and_0 : std_logic_vector(1 downto 0);
	signal entry_port_and_1 : std_logic_vector(1 downto 0);
	signal entry_port_hs_0 : std_logic_vector(1 downto 0);
	signal entry_port_hs_1 : std_logic_vector(1 downto 0);
	signal TEMP_3_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_in_1 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_1 : std_logic_vector(3 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_0, entry_port_idx_0)
	entry_port_valid_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_valid_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_1, entry_port_idx_1)
	entry_port_valid_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_valid_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_bits_0, port_bits, entry_port_valid_0)
	TEMP_1_mux_0 <= port_bits_0_i when entry_port_valid_0(0) = '1' else "0000000000";
	TEMP_1_mux_1 <= port_bits_1_i when entry_port_valid_0(1) = '1' else "0000000000";
	entry_bits_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_bits_1, port_bits, entry_port_valid_1)
	TEMP_2_mux_0 <= port_bits_0_i when entry_port_valid_1(0) = '1' else "0000000000";
	TEMP_2_mux_1 <= port_bits_1_i when entry_port_valid_1(1) = '1' else "0000000000";
	entry_bits_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	entry_request_valid_0 <= entry_valid_0_i and not entry_bits_valid_0_i;
	entry_request_valid_1 <= entry_valid_1_i and not entry_bits_valid_1_i;
	entry_port_request_0 <= entry_port_valid_0 when entry_request_valid_0 else "00";
	entry_port_request_1 <= entry_port_valid_1 when entry_request_valid_1 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_port_request, or)
	port_ready_vec <= entry_port_request_0 or entry_port_request_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	port_ready_1_o <= port_ready_vec(1);
	entry_port_and_0(0) <= entry_port_request_0(0) and port_valid_0_i;
	entry_port_and_0(1) <= entry_port_request_0(1) and port_valid_1_i;
	entry_port_and_1(0) <= entry_port_request_1(0) and port_valid_0_i;
	entry_port_and_1(1) <= entry_port_request_1(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_hs, entry_port_and, queue_head_oh)
	TEMP_3_double_in_0(0) <= entry_port_and_0(0);
	TEMP_3_double_in_0(2) <= entry_port_and_0(0);
	TEMP_3_double_in_0(1) <= entry_port_and_1(0);
	TEMP_3_double_in_0(3) <= entry_port_and_1(0);
	TEMP_3_double_out_0 <= TEMP_3_double_in_0 and not std_logic_vector( unsigned( TEMP_3_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(0) <= TEMP_3_double_out_0(0) or TEMP_3_double_out_0(2);
	entry_port_hs_1(0) <= TEMP_3_double_out_0(1) or TEMP_3_double_out_0(3);
	TEMP_3_double_in_1(0) <= entry_port_and_0(1);
	TEMP_3_double_in_1(2) <= entry_port_and_0(1);
	TEMP_3_double_in_1(1) <= entry_port_and_1(1);
	TEMP_3_double_in_1(3) <= entry_port_and_1(1);
	TEMP_3_double_out_1 <= TEMP_3_double_in_1 and not std_logic_vector( unsigned( TEMP_3_double_in_1 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(1) <= TEMP_3_double_out_1(0) or TEMP_3_double_out_1(2);
	entry_port_hs_1(1) <= TEMP_3_double_out_1(1) or TEMP_3_double_out_1(3);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_hs_0, or)
	entry_wen_0_o <= entry_port_hs_0(0) or entry_port_hs_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_hs_1, or)
	entry_wen_1_o <= entry_port_hs_1(0) or entry_port_hs_1(1);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core_std is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_bits_0_i : in std_logic_vector(12 downto 0);
		port_bits_1_i : in std_logic_vector(12 downto 0);
		port_valid_0_i : in std_logic;
		port_valid_1_i : in std_logic;
		port_ready_0_o : out std_logic;
		port_ready_1_o : out std_logic;
		entry_valid_0_i : in std_logic;
		entry_valid_1_i : in std_logic;
		entry_bits_valid_0_i : in std_logic;
		entry_bits_valid_1_i : in std_logic;
		entry_port_idx_0_i : in std_logic_vector(0 downto 0);
		entry_port_idx_1_i : in std_logic_vector(0 downto 0);
		entry_bits_0_o : out std_logic_vector(12 downto 0);
		entry_bits_1_o : out std_logic_vector(12 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(1 downto 0)
	);
end entity;

architecture arch of config_0_core_std is
	signal entry_port_valid_0 : std_logic_vector(1 downto 0);
	signal entry_port_valid_1 : std_logic_vector(1 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_1_mux_1 : std_logic_vector(12 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(12 downto 0);
	signal entry_request_valid_0 : std_logic;
	signal entry_request_valid_1 : std_logic;
	signal entry_port_request_0 : std_logic_vector(1 downto 0);
	signal entry_port_request_1 : std_logic_vector(1 downto 0);
	signal port_ready_vec : std_logic_vector(1 downto 0);
	signal entry_port_and_0 : std_logic_vector(1 downto 0);
	signal entry_port_and_1 : std_logic_vector(1 downto 0);
	signal entry_port_hs_0 : std_logic_vector(1 downto 0);
	signal entry_port_hs_1 : std_logic_vector(1 downto 0);
	signal TEMP_3_double_in_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_in_1 : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out_1 : std_logic_vector(3 downto 0);
begin
	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_0, entry_port_idx_0)
	entry_port_valid_0(0) <= '1' when entry_port_idx_0_i = "0" else '0';
	entry_port_valid_0(1) <= '1' when entry_port_idx_0_i = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(entry_port_valid_1, entry_port_idx_1)
	entry_port_valid_1(0) <= '1' when entry_port_idx_1_i = "0" else '0';
	entry_port_valid_1(1) <= '1' when entry_port_idx_1_i = "1" else '0';
	-- Bits To One-Hot End

	-- Mux1H Begin
	-- Mux1H(entry_bits_0, port_bits, entry_port_valid_0)
	TEMP_1_mux_0 <= port_bits_0_i when entry_port_valid_0(0) = '1' else "0000000000000";
	TEMP_1_mux_1 <= port_bits_1_i when entry_port_valid_0(1) = '1' else "0000000000000";
	entry_bits_0_o <= TEMP_1_mux_0 or TEMP_1_mux_1;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_bits_1, port_bits, entry_port_valid_1)
	TEMP_2_mux_0 <= port_bits_0_i when entry_port_valid_1(0) = '1' else "0000000000000";
	TEMP_2_mux_1 <= port_bits_1_i when entry_port_valid_1(1) = '1' else "0000000000000";
	entry_bits_1_o <= TEMP_2_mux_0 or TEMP_2_mux_1;
	-- Mux1H End

	entry_request_valid_0 <= entry_valid_0_i and not entry_bits_valid_0_i;
	entry_request_valid_1 <= entry_valid_1_i and not entry_bits_valid_1_i;
	entry_port_request_0 <= entry_port_valid_0 when entry_request_valid_0 else "00";
	entry_port_request_1 <= entry_port_valid_1 when entry_request_valid_1 else "00";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_port_request, or)
	port_ready_vec <= entry_port_request_0 or entry_port_request_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	port_ready_1_o <= port_ready_vec(1);
	entry_port_and_0(0) <= entry_port_request_0(0) and port_valid_0_i;
	entry_port_and_0(1) <= entry_port_request_0(1) and port_valid_1_i;
	entry_port_and_1(0) <= entry_port_request_1(0) and port_valid_0_i;
	entry_port_and_1(1) <= entry_port_request_1(1) and port_valid_1_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_hs, entry_port_and, queue_head_oh)
	TEMP_3_double_in_0(0) <= entry_port_and_0(0);
	TEMP_3_double_in_0(2) <= entry_port_and_0(0);
	TEMP_3_double_in_0(1) <= entry_port_and_1(0);
	TEMP_3_double_in_0(3) <= entry_port_and_1(0);
	TEMP_3_double_out_0 <= TEMP_3_double_in_0 and not std_logic_vector( unsigned( TEMP_3_double_in_0 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(0) <= TEMP_3_double_out_0(0) or TEMP_3_double_out_0(2);
	entry_port_hs_1(0) <= TEMP_3_double_out_0(1) or TEMP_3_double_out_0(3);
	TEMP_3_double_in_1(0) <= entry_port_and_0(1);
	TEMP_3_double_in_1(2) <= entry_port_and_0(1);
	TEMP_3_double_in_1(1) <= entry_port_and_1(1);
	TEMP_3_double_in_1(3) <= entry_port_and_1(1);
	TEMP_3_double_out_1 <= TEMP_3_double_in_1 and not std_logic_vector( unsigned( TEMP_3_double_in_1 ) - unsigned( "00" & queue_head_oh_i ) );
	entry_port_hs_0(1) <= TEMP_3_double_out_1(0) or TEMP_3_double_out_1(2);
	entry_port_hs_1(1) <= TEMP_3_double_out_1(1) or TEMP_3_double_out_1(3);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_hs_0, or)
	entry_wen_0_o <= entry_port_hs_0(0) or entry_port_hs_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_hs_1, or)
	entry_wen_1_o <= entry_port_hs_1(0) or entry_port_hs_1(1);
	-- Reduction End


end architecture;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity config_0_core is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldp_addr_0_i : in std_logic_vector(9 downto 0);
		ldp_addr_1_i : in std_logic_vector(9 downto 0);
		ldp_addr_valid_0_i : in std_logic;
		ldp_addr_valid_1_i : in std_logic;
		ldp_addr_ready_0_o : out std_logic;
		ldp_addr_ready_1_o : out std_logic;
		ldp_data_0_o : out std_logic_vector(12 downto 0);
		ldp_data_1_o : out std_logic_vector(12 downto 0);
		ldp_data_valid_0_o : out std_logic;
		ldp_data_valid_1_o : out std_logic;
		ldp_data_ready_0_i : in std_logic;
		ldp_data_ready_1_i : in std_logic;
		stp_addr_0_i : in std_logic_vector(9 downto 0);
		stp_addr_1_i : in std_logic_vector(9 downto 0);
		stp_addr_valid_0_i : in std_logic;
		stp_addr_valid_1_i : in std_logic;
		stp_addr_ready_0_o : out std_logic;
		stp_addr_ready_1_o : out std_logic;
		stp_data_0_i : in std_logic_vector(12 downto 0);
		stp_data_1_i : in std_logic_vector(12 downto 0);
		stp_data_valid_0_i : in std_logic;
		stp_data_valid_1_i : in std_logic;
		stp_data_ready_0_o : out std_logic;
		stp_data_ready_1_o : out std_logic;
		empty_o : out std_logic;
		rreq_valid_0_o : out std_logic;
		rreq_ready_0_i : in std_logic;
		rreq_id_0_o : out std_logic_vector(0 downto 0);
		rreq_addr_0_o : out std_logic_vector(9 downto 0);
		rresp_valid_0_i : in std_logic;
		rresp_ready_0_o : out std_logic;
		rresp_id_0_i : in std_logic_vector(0 downto 0);
		rresp_data_0_i : in std_logic_vector(12 downto 0);
		wreq_valid_0_o : out std_logic;
		wreq_ready_0_i : in std_logic;
		wreq_id_0_o : out std_logic_vector(0 downto 0);
		wreq_addr_0_o : out std_logic_vector(9 downto 0);
		wreq_data_0_o : out std_logic_vector(12 downto 0);
		wresp_valid_0_i : in std_logic;
		wresp_ready_0_o : out std_logic;
		wresp_id_0_i : in std_logic_vector(0 downto 0);
		memStart_ready_o : out std_logic;
		memStart_valid_i : in std_logic;
		ctrlEnd_ready_o : out std_logic;
		ctrlEnd_valid_i : in std_logic;
		memEnd_ready_i : in std_logic;
		memEnd_valid_o : out std_logic
	);
end entity;

architecture arch of config_0_core is
	signal memStartReady : std_logic;
	signal memEndValid : std_logic;
	signal ctrlEndReady : std_logic;
	signal TEMP_GEN_MEM : std_logic;
	signal ldq_valid_0_d : std_logic;
	signal ldq_valid_0_q : std_logic;
	signal ldq_valid_1_d : std_logic;
	signal ldq_valid_1_q : std_logic;
	signal ldq_issue_0_d : std_logic;
	signal ldq_issue_0_q : std_logic;
	signal ldq_issue_1_d : std_logic;
	signal ldq_issue_1_q : std_logic;
	signal ldq_port_idx_0_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_0_q : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_d : std_logic_vector(0 downto 0);
	signal ldq_port_idx_1_q : std_logic_vector(0 downto 0);
	signal ldq_addr_valid_0_d : std_logic;
	signal ldq_addr_valid_0_q : std_logic;
	signal ldq_addr_valid_1_d : std_logic;
	signal ldq_addr_valid_1_q : std_logic;
	signal ldq_addr_0_d : std_logic_vector(9 downto 0);
	signal ldq_addr_0_q : std_logic_vector(9 downto 0);
	signal ldq_addr_1_d : std_logic_vector(9 downto 0);
	signal ldq_addr_1_q : std_logic_vector(9 downto 0);
	signal ldq_data_valid_0_d : std_logic;
	signal ldq_data_valid_0_q : std_logic;
	signal ldq_data_valid_1_d : std_logic;
	signal ldq_data_valid_1_q : std_logic;
	signal ldq_data_0_d : std_logic_vector(12 downto 0);
	signal ldq_data_0_q : std_logic_vector(12 downto 0);
	signal ldq_data_1_d : std_logic_vector(12 downto 0);
	signal ldq_data_1_q : std_logic_vector(12 downto 0);
	signal stq_valid_0_d : std_logic;
	signal stq_valid_0_q : std_logic;
	signal stq_valid_1_d : std_logic;
	signal stq_valid_1_q : std_logic;
	signal stq_port_idx_0_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_0_q : std_logic_vector(0 downto 0);
	signal stq_port_idx_1_d : std_logic_vector(0 downto 0);
	signal stq_port_idx_1_q : std_logic_vector(0 downto 0);
	signal stq_addr_valid_0_d : std_logic;
	signal stq_addr_valid_0_q : std_logic;
	signal stq_addr_valid_1_d : std_logic;
	signal stq_addr_valid_1_q : std_logic;
	signal stq_addr_0_d : std_logic_vector(9 downto 0);
	signal stq_addr_0_q : std_logic_vector(9 downto 0);
	signal stq_addr_1_d : std_logic_vector(9 downto 0);
	signal stq_addr_1_q : std_logic_vector(9 downto 0);
	signal stq_data_valid_0_d : std_logic;
	signal stq_data_valid_0_q : std_logic;
	signal stq_data_valid_1_d : std_logic;
	signal stq_data_valid_1_q : std_logic;
	signal stq_data_0_d : std_logic_vector(12 downto 0);
	signal stq_data_0_q : std_logic_vector(12 downto 0);
	signal stq_data_1_d : std_logic_vector(12 downto 0);
	signal stq_data_1_q : std_logic_vector(12 downto 0);
	signal store_is_older_0_d : std_logic_vector(1 downto 0);
	signal store_is_older_0_q : std_logic_vector(1 downto 0);
	signal store_is_older_1_d : std_logic_vector(1 downto 0);
	signal store_is_older_1_q : std_logic_vector(1 downto 0);
	signal ldq_tail_d : std_logic_vector(0 downto 0);
	signal ldq_tail_q : std_logic_vector(0 downto 0);
	signal ldq_head_d : std_logic_vector(0 downto 0);
	signal ldq_head_q : std_logic_vector(0 downto 0);
	signal stq_tail_d : std_logic_vector(0 downto 0);
	signal stq_tail_q : std_logic_vector(0 downto 0);
	signal stq_head_d : std_logic_vector(0 downto 0);
	signal stq_head_q : std_logic_vector(0 downto 0);
	signal stq_issue_d : std_logic_vector(0 downto 0);
	signal stq_issue_q : std_logic_vector(0 downto 0);
	signal stq_resp_d : std_logic_vector(0 downto 0);
	signal stq_resp_q : std_logic_vector(0 downto 0);
	signal ldq_wen_0 : std_logic;
	signal ldq_wen_1 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(1 downto 0);
	signal ga_ls_order_1 : std_logic_vector(1 downto 0);
	signal num_loads : std_logic_vector(0 downto 0);
	signal num_stores : std_logic_vector(0 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(1 downto 0);
	signal stq_head_oh : std_logic_vector(1 downto 0);
	signal ldq_valid_next_0 : std_logic;
	signal ldq_valid_next_1 : std_logic;
	signal stq_valid_next_0 : std_logic;
	signal stq_valid_next_1 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal ldq_tail_oh : std_logic_vector(1 downto 0);
	signal ldq_head_next_oh : std_logic_vector(1 downto 0);
	signal ldq_head_next : std_logic_vector(0 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_1_double_in : std_logic_vector(3 downto 0);
	signal TEMP_1_double_out : std_logic_vector(3 downto 0);
	signal TEMP_2_in_0_0 : std_logic;
	signal TEMP_2_in_0_1 : std_logic;
	signal TEMP_2_out_0 : std_logic;
	signal stq_tail_oh : std_logic_vector(1 downto 0);
	signal stq_head_next_oh : std_logic_vector(1 downto 0);
	signal stq_head_next : std_logic_vector(0 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(1 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(0 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(1 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(1 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(1 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(1 downto 0);
	signal can_bypass_0 : std_logic_vector(1 downto 0);
	signal can_bypass_1 : std_logic_vector(1 downto 0);
	signal addr_valid_0 : std_logic_vector(1 downto 0);
	signal addr_valid_1 : std_logic_vector(1 downto 0);
	signal addr_same_0 : std_logic_vector(1 downto 0);
	signal addr_same_1 : std_logic_vector(1 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal TEMP_3_double_in : std_logic_vector(3 downto 0);
	signal TEMP_3_double_out : std_logic_vector(3 downto 0);
	signal st_ld_conflict : std_logic_vector(1 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal stq_last_oh : std_logic_vector(1 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(1 downto 0);
	signal TEMP_4_double_in : std_logic_vector(3 downto 0);
	signal TEMP_4_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_4_double_out : std_logic_vector(3 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(1 downto 0);
	signal TEMP_5_double_in : std_logic_vector(3 downto 0);
	signal TEMP_5_base_rev : std_logic_vector(1 downto 0);
	signal TEMP_5_double_out : std_logic_vector(3 downto 0);
	signal TEMP_6_in_0_0 : std_logic;
	signal TEMP_6_in_0_1 : std_logic;
	signal TEMP_6_out_0 : std_logic;
	signal TEMP_7_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_7_mux_1 : std_logic_vector(9 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(12 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(12 downto 0);
	signal bypass_data_0 : std_logic_vector(12 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_9_mux_1 : std_logic_vector(12 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(12 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(12 downto 0);
	signal bypass_data_1 : std_logic_vector(12 downto 0);
	signal TEMP_11_mux_0 : std_logic_vector(12 downto 0);
	signal TEMP_11_mux_1 : std_logic_vector(12 downto 0);
begin
	-- Define the intermediate logic
	TEMP_GEN_MEM <= ctrlEnd_valid_i and stq_empty and ldq_empty;
	-- Define logic for the new interfaces needed by dynamatic
	process (clk) is
	begin
		if rising_edge(clk) then
			if rst = '1' then
				memStartReady <= '1';
				memEndValid <= '0';
				ctrlEndReady <= '0';
			else
				memStartReady <= (memEndValid and memEnd_ready_i) or ((not (memStart_valid_i and memStartReady)) and memStartReady);
				memEndValid <= TEMP_GEN_MEM or memEndValid;
				ctrlEndReady <= (not (ctrlEnd_valid_i and ctrlEndReady)) and (TEMP_GEN_MEM or ctrlEndReady);
			end if;
		end if;
	end process;

	-- Update new memory interfaces
	memStart_ready_o <= memStartReady;
	ctrlEnd_ready_o <= ctrlEndReady;
	memEnd_valid_o <= memEndValid;
	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_head_oh, ldq_head)
	ldq_head_oh(0) <= '1' when ldq_head_q = "0" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "1" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "0" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "1" else '0';
	-- Bits To One-Hot End

	ldq_valid_next_0 <= not ldq_reset_0 and ldq_valid_0_q;
	ldq_valid_0_d <= ldq_wen_0 or ldq_valid_next_0;
	ldq_issue_0_d <= not ldq_wen_0 and ( ldq_issue_set_0 or ldq_issue_0_q );
	ldq_addr_valid_0_d <= not ldq_wen_0 and ( ldq_addr_wen_0 or ldq_addr_valid_0_q );
	ldq_data_valid_0_d <= not ldq_wen_0 and ( ldq_data_wen_0 or ldq_data_valid_0_q );
	ldq_valid_next_1 <= not ldq_reset_1 and ldq_valid_1_q;
	ldq_valid_1_d <= ldq_wen_1 or ldq_valid_next_1;
	ldq_issue_1_d <= not ldq_wen_1 and ( ldq_issue_set_1 or ldq_issue_1_q );
	ldq_addr_valid_1_d <= not ldq_wen_1 and ( ldq_addr_wen_1 or ldq_addr_valid_1_q );
	ldq_data_valid_1_d <= not ldq_wen_1 and ( ldq_data_wen_1 or ldq_data_valid_1_q );
	stq_valid_next_0 <= not stq_reset_0 and stq_valid_0_q;
	stq_valid_0_d <= stq_wen_0 or stq_valid_next_0;
	stq_addr_valid_0_d <= not stq_wen_0 and ( stq_addr_wen_0 or stq_addr_valid_0_q );
	stq_data_valid_0_d <= not stq_wen_0 and ( stq_data_wen_0 or stq_data_valid_0_q );
	stq_valid_next_1 <= not stq_reset_1 and stq_valid_1_q;
	stq_valid_1_d <= stq_wen_1 or stq_valid_next_1;
	stq_addr_valid_1_d <= not stq_wen_1 and ( stq_addr_wen_1 or stq_addr_valid_1_q );
	stq_data_valid_1_d <= not stq_wen_1 and ( stq_data_wen_1 or stq_data_valid_1_q );
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_valid_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_valid_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_valid_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_valid_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_valid, or)
	ldq_not_empty <= ldq_valid_0_q or ldq_valid_1_q;
	-- Reduction End

	ldq_empty <= not ldq_not_empty;
	-- MuxLookUp Begin
	-- MuxLookUp(stq_not_empty, stq_valid, stq_head)
	stq_not_empty <= 
	stq_valid_0_q when (stq_head_q = "0") else
	stq_valid_1_q when (stq_head_q = "1") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 2)
	ldq_tail_d <= std_logic_vector(unsigned(ldq_tail_q) + unsigned(num_loads));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 2)
	stq_tail_d <= std_logic_vector(unsigned(stq_tail_q) + unsigned(num_stores));
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 2)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 2)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "0" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "1" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_valid_next, ldq_tail_oh)
	TEMP_1_double_in(0) <= ldq_valid_next_0;
	TEMP_1_double_in(2) <= ldq_valid_next_0;
	TEMP_1_double_in(1) <= ldq_valid_next_1;
	TEMP_1_double_in(3) <= ldq_valid_next_1;
	TEMP_1_double_out <= TEMP_1_double_in and not std_logic_vector( unsigned( TEMP_1_double_in ) - unsigned( "00" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_1_double_out(1 downto 0) or TEMP_1_double_out(3 downto 2);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_valid_next, or)
	ldq_head_sel <= ldq_valid_next_0 or ldq_valid_next_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_2_in_0_0 <= '0';
	TEMP_2_in_0_1 <= ldq_head_next_oh(1);
	TEMP_2_out_0 <= TEMP_2_in_0_0 or TEMP_2_in_0_1;
	ldq_head_next(0) <= TEMP_2_out_0;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "0" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "1" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 2)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) + 1);
	-- WrapAdd End

	stq_head_sel <= wresp_valid_0_i;
	stq_head_d <= stq_head_next when stq_head_sel else stq_head_q;
	config_0_core_ga : entity work.config_0_core_ga
		port map(
			rst => rst,
			clk => clk,
			group_init_valid_0_i => group_init_valid_0_i,
			group_init_ready_0_o => group_init_ready_0_o,
			ldq_tail_i => ldq_tail_q,
			ldq_head_i => ldq_head_q,
			ldq_empty_i => ldq_empty,
			stq_tail_i => stq_tail_q,
			stq_head_i => stq_head_q,
			stq_empty_i => stq_empty,
			ldq_wen_0_o => ldq_wen_0,
			ldq_wen_1_o => ldq_wen_1,
			num_loads_o => num_loads,
			ldq_port_idx_0_o => ldq_port_idx_0_d,
			ldq_port_idx_1_o => ldq_port_idx_1_d,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			stq_port_idx_0_o => stq_port_idx_0_d,
			stq_port_idx_1_o => stq_port_idx_1_d,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			num_stores_o => num_stores
		);
	config_0_core_lda_dispatcher : entity work.config_0_core_lda
		port map(
			rst => rst,
			clk => clk,
			port_bits_0_i => ldp_addr_0_i,
			port_bits_1_i => ldp_addr_1_i,
			port_ready_0_o => ldp_addr_ready_0_o,
			port_ready_1_o => ldp_addr_ready_1_o,
			port_valid_0_i => ldp_addr_valid_0_i,
			port_valid_1_i => ldp_addr_valid_1_i,
			entry_valid_0_i => ldq_valid_0_q,
			entry_valid_1_i => ldq_valid_1_q,
			entry_bits_valid_0_i => ldq_addr_valid_0_q,
			entry_bits_valid_1_i => ldq_addr_valid_1_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_bits_0_o => ldq_addr_0_d,
			entry_bits_1_o => ldq_addr_1_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			queue_head_oh_i => ldq_head_oh
		);
	config_0_core_ldd_dispatcher : entity work.config_0_core_ldd
		port map(
			rst => rst,
			clk => clk,
			port_bits_0_o => ldp_data_0_o,
			port_bits_1_o => ldp_data_1_o,
			port_ready_0_i => ldp_data_ready_0_i,
			port_ready_1_i => ldp_data_ready_1_i,
			port_valid_0_o => ldp_data_valid_0_o,
			port_valid_1_o => ldp_data_valid_1_o,
			entry_valid_0_i => ldq_valid_0_q,
			entry_valid_1_i => ldq_valid_1_q,
			entry_bits_valid_0_i => ldq_data_valid_0_q,
			entry_bits_valid_1_i => ldq_data_valid_1_q,
			entry_port_idx_0_i => ldq_port_idx_0_q,
			entry_port_idx_1_i => ldq_port_idx_1_q,
			entry_bits_0_i => ldq_data_0_q,
			entry_bits_1_i => ldq_data_1_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
			queue_head_oh_i => ldq_head_oh
		);
	config_0_core_sta_dispatcher : entity work.config_0_core_sta
		port map(
			rst => rst,
			clk => clk,
			port_bits_0_i => stp_addr_0_i,
			port_bits_1_i => stp_addr_1_i,
			port_ready_0_o => stp_addr_ready_0_o,
			port_ready_1_o => stp_addr_ready_1_o,
			port_valid_0_i => stp_addr_valid_0_i,
			port_valid_1_i => stp_addr_valid_1_i,
			entry_valid_0_i => stq_valid_0_q,
			entry_valid_1_i => stq_valid_1_q,
			entry_bits_valid_0_i => stq_addr_valid_0_q,
			entry_bits_valid_1_i => stq_addr_valid_1_q,
			entry_port_idx_0_i => stq_port_idx_0_q,
			entry_port_idx_1_i => stq_port_idx_1_q,
			entry_bits_0_o => stq_addr_0_d,
			entry_bits_1_o => stq_addr_1_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			queue_head_oh_i => stq_head_oh
		);
	config_0_core_std_dispatcher : entity work.config_0_core_std
		port map(
			rst => rst,
			clk => clk,
			port_bits_0_i => stp_data_0_i,
			port_bits_1_i => stp_data_1_i,
			port_ready_0_o => stp_data_ready_0_o,
			port_ready_1_o => stp_data_ready_1_o,
			port_valid_0_i => stp_data_valid_0_i,
			port_valid_1_i => stp_data_valid_1_i,
			entry_valid_0_i => stq_valid_0_q,
			entry_valid_1_i => stq_valid_1_q,
			entry_bits_valid_0_i => stq_data_valid_0_q,
			entry_bits_valid_1_i => stq_data_valid_1_q,
			entry_port_idx_0_i => stq_port_idx_0_q,
			entry_port_idx_1_i => stq_port_idx_1_q,
			entry_bits_0_o => stq_data_0_d,
			entry_bits_1_o => stq_data_1_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			queue_head_oh_i => stq_head_oh
		);
	addr_valid_0(0) <= ldq_addr_valid_0_q and stq_addr_valid_0_q;
	addr_valid_0(1) <= ldq_addr_valid_0_q and stq_addr_valid_1_q;
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	ld_st_conflict_0(0) <= stq_valid_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_valid_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(0) <= stq_valid_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_valid_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	can_bypass_0(0) <= ldq_valid_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_valid_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_1(0) <= ldq_valid_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_valid_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	load_conflict_0 <= ld_st_conflict_0(0) or ld_st_conflict_0(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	load_conflict_1 <= ld_st_conflict_1(0) or ld_st_conflict_1(1);
	-- Reduction End

	load_req_valid_0 <= ldq_valid_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_valid_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_3_double_in(0) <= can_load_0;
	TEMP_3_double_in(2) <= can_load_0;
	TEMP_3_double_in(1) <= can_load_1;
	TEMP_3_double_in(3) <= can_load_1;
	TEMP_3_double_out <= TEMP_3_double_in and not std_logic_vector( unsigned( TEMP_3_double_in ) - unsigned( "00" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_3_double_out(1 downto 0) or TEMP_3_double_out(3 downto 2);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	load_en_0 <= can_load_0 or can_load_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_valid_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_valid_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	store_conflict <= st_ld_conflict(0) or st_ld_conflict(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_valid, stq_issue)
	store_valid <= 
	stq_valid_0_q when (stq_issue_q = "0") else
	stq_valid_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
	store_data_valid <= 
	stq_data_valid_0_q when (stq_issue_q = "0") else
	stq_data_valid_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
	store_addr_valid <= 
	stq_addr_valid_0_q when (stq_issue_q = "0") else
	stq_addr_valid_1_q when (stq_issue_q = "1") else
	'0';
	-- MuxLookUp End

	store_en <= not store_conflict and store_valid and store_data_valid and store_addr_valid;
	store_idx <= stq_issue_q;
	-- Bits To One-Hot Begin
	-- BitsToOHSub1(stq_last_oh, stq_tail)
	stq_last_oh(0) <= '1' when stq_tail_q = "1" else '0';
	stq_last_oh(1) <= '1' when stq_tail_q = "0" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_4_double_in(0) <= ld_st_conflict_0(1);
	TEMP_4_double_in(2) <= ld_st_conflict_0(1);
	TEMP_4_double_in(1) <= ld_st_conflict_0(0);
	TEMP_4_double_in(3) <= ld_st_conflict_0(0);
	TEMP_4_base_rev(0) <= stq_last_oh(1);
	TEMP_4_base_rev(1) <= stq_last_oh(0);
	TEMP_4_double_out <= TEMP_4_double_in and not std_logic_vector( unsigned( TEMP_4_double_in ) - unsigned( "00" & TEMP_4_base_rev ) );
	bypass_idx_oh_0(1) <= TEMP_4_double_out(0) or TEMP_4_double_out(2);
	bypass_idx_oh_0(0) <= TEMP_4_double_out(1) or TEMP_4_double_out(3);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	bypass_en_0 <= bypass_en_vec_0(0) or bypass_en_vec_0(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_5_double_in(0) <= ld_st_conflict_1(1);
	TEMP_5_double_in(2) <= ld_st_conflict_1(1);
	TEMP_5_double_in(1) <= ld_st_conflict_1(0);
	TEMP_5_double_in(3) <= ld_st_conflict_1(0);
	TEMP_5_base_rev(0) <= stq_last_oh(1);
	TEMP_5_base_rev(1) <= stq_last_oh(0);
	TEMP_5_double_out <= TEMP_5_double_in and not std_logic_vector( unsigned( TEMP_5_double_in ) - unsigned( "00" & TEMP_5_base_rev ) );
	bypass_idx_oh_1(1) <= TEMP_5_double_out(0) or TEMP_5_double_out(2);
	bypass_idx_oh_1(0) <= TEMP_5_double_out(1) or TEMP_5_double_out(3);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	bypass_en_1 <= bypass_en_vec_1(0) or bypass_en_vec_1(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_6_in_0_0 <= '0';
	TEMP_6_in_0_1 <= load_idx_oh_0(1);
	TEMP_6_out_0 <= TEMP_6_in_0_0 or TEMP_6_in_0_1;
	rreq_id_0_o(0) <= TEMP_6_out_0;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_7_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "0000000000";
	TEMP_7_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "0000000000";
	rreq_addr_0_o <= TEMP_7_mux_0 or TEMP_7_mux_1;
	-- Mux1H End

	ldq_issue_set_vec_0(0) <= ( load_idx_oh_0(0) and rreq_ready_0_i and load_en_0 ) or bypass_en_0;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_0, ldq_issue_set_vec_0, or)
	ldq_issue_set_0 <= ldq_issue_set_vec_0(0);
	-- Reduction End

	ldq_issue_set_vec_1(0) <= ( load_idx_oh_0(1) and rreq_ready_0_i and load_en_0 ) or bypass_en_1;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_1, ldq_issue_set_vec_1, or)
	ldq_issue_set_1 <= ldq_issue_set_vec_1(0);
	-- Reduction End

	wreq_valid_0_o <= store_en;
	wreq_id_0_o <= "0";
	-- MuxLookUp Begin
	-- MuxLookUp(wreq_addr_0, stq_addr, store_idx)
	wreq_addr_0_o <= 
	stq_addr_0_q when (store_idx = "0") else
	stq_addr_1_q when (store_idx = "1") else
	"0000000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "0") else
	stq_data_1_q when (store_idx = "1") else
	"0000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_8_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "0000000000000";
	read_data_0 <= TEMP_8_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_9_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "0000000000000";
	TEMP_9_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "0000000000000";
	bypass_data_0 <= TEMP_9_mux_0 or TEMP_9_mux_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_10_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "0000000000000";
	read_data_1 <= TEMP_10_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_11_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "0000000000000";
	TEMP_11_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "0000000000000";
	bypass_data_1 <= TEMP_11_mux_0 or TEMP_11_mux_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "0" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "1" ) else '0';
	stq_resp_en <= wresp_valid_0_i;
	wresp_ready_0_o <= '1';

	process (clk, rst) is
	begin
		if (rst = '1') then
			ldq_valid_0_q <= '0';
			ldq_valid_1_q <= '0';
		elsif (rising_edge(clk)) then
			ldq_valid_0_q <= ldq_valid_0_d;
			ldq_valid_1_q <= ldq_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			ldq_issue_0_q <= ldq_issue_0_d;
			ldq_issue_1_q <= ldq_issue_1_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_wen_0 = '1') then
				ldq_port_idx_0_q <= ldq_port_idx_0_d;
			end if;
			if (ldq_wen_1 = '1') then
				ldq_port_idx_1_q <= ldq_port_idx_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_addr_valid_0_q <= ldq_addr_valid_0_d;
			ldq_addr_valid_1_q <= ldq_addr_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_addr_wen_0 = '1') then
				ldq_addr_0_q <= ldq_addr_0_d;
			end if;
			if (ldq_addr_wen_1 = '1') then
				ldq_addr_1_q <= ldq_addr_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_data_valid_0_q <= ldq_data_valid_0_d;
			ldq_data_valid_1_q <= ldq_data_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_data_wen_0 = '1') then
				ldq_data_0_q <= ldq_data_0_d;
			end if;
			if (ldq_data_wen_1 = '1') then
				ldq_data_1_q <= ldq_data_1_d;
			end if;
		end if;
		if (rst = '1') then
			stq_valid_0_q <= '0';
			stq_valid_1_q <= '0';
		elsif (rising_edge(clk)) then
			stq_valid_0_q <= stq_valid_0_d;
			stq_valid_1_q <= stq_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_wen_0 = '1') then
				stq_port_idx_0_q <= stq_port_idx_0_d;
			end if;
			if (stq_wen_1 = '1') then
				stq_port_idx_1_q <= stq_port_idx_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_addr_valid_0_q <= stq_addr_valid_0_d;
			stq_addr_valid_1_q <= stq_addr_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_addr_wen_0 = '1') then
				stq_addr_0_q <= stq_addr_0_d;
			end if;
			if (stq_addr_wen_1 = '1') then
				stq_addr_1_q <= stq_addr_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_data_valid_0_q <= stq_data_valid_0_d;
			stq_data_valid_1_q <= stq_data_valid_1_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_data_wen_0 = '1') then
				stq_data_0_q <= stq_data_0_d;
			end if;
			if (stq_data_wen_1 = '1') then
				stq_data_1_q <= stq_data_1_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			store_is_older_0_q <= store_is_older_0_d;
			store_is_older_1_q <= store_is_older_1_d;
		end if;
		if (rst = '1') then
			ldq_tail_q <= "0";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "0";
		elsif (rising_edge(clk)) then
			ldq_head_q <= ldq_head_d;
		end if;
		if (rst = '1') then
			stq_tail_q <= "0";
		elsif (rising_edge(clk)) then
			stq_tail_q <= stq_tail_d;
		end if;
		if (rst = '1') then
			stq_head_q <= "0";
		elsif (rising_edge(clk)) then
			stq_head_q <= stq_head_d;
		end if;
		if (rst = '1') then
			stq_issue_q <= "0";
		elsif (rising_edge(clk)) then
			if (stq_issue_en = '1') then
				stq_issue_q <= stq_issue_d;
			end if;
		end if;
		if (rst = '1') then
			stq_resp_q <= "0";
		elsif (rising_edge(clk)) then
			if (stq_resp_en = '1') then
				stq_resp_q <= stq_resp_d;
			end if;
		end if;
	end process;
end architecture;
