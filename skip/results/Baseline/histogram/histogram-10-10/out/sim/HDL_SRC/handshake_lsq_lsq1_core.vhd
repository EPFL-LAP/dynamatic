

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ga is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldq_tail_i : in std_logic_vector(3 downto 0);
		ldq_head_i : in std_logic_vector(3 downto 0);
		ldq_empty_i : in std_logic;
		stq_tail_i : in std_logic_vector(3 downto 0);
		stq_head_i : in std_logic_vector(3 downto 0);
		stq_empty_i : in std_logic;
		ldq_wen_0_o : out std_logic;
		ldq_wen_1_o : out std_logic;
		ldq_wen_2_o : out std_logic;
		ldq_wen_3_o : out std_logic;
		ldq_wen_4_o : out std_logic;
		ldq_wen_5_o : out std_logic;
		ldq_wen_6_o : out std_logic;
		ldq_wen_7_o : out std_logic;
		ldq_wen_8_o : out std_logic;
		ldq_wen_9_o : out std_logic;
		num_loads_o : out std_logic_vector(3 downto 0);
		stq_wen_0_o : out std_logic;
		stq_wen_1_o : out std_logic;
		stq_wen_2_o : out std_logic;
		stq_wen_3_o : out std_logic;
		stq_wen_4_o : out std_logic;
		stq_wen_5_o : out std_logic;
		stq_wen_6_o : out std_logic;
		stq_wen_7_o : out std_logic;
		stq_wen_8_o : out std_logic;
		stq_wen_9_o : out std_logic;
		num_stores_o : out std_logic_vector(3 downto 0);
		ga_ls_order_0_o : out std_logic_vector(9 downto 0);
		ga_ls_order_1_o : out std_logic_vector(9 downto 0);
		ga_ls_order_2_o : out std_logic_vector(9 downto 0);
		ga_ls_order_3_o : out std_logic_vector(9 downto 0);
		ga_ls_order_4_o : out std_logic_vector(9 downto 0);
		ga_ls_order_5_o : out std_logic_vector(9 downto 0);
		ga_ls_order_6_o : out std_logic_vector(9 downto 0);
		ga_ls_order_7_o : out std_logic_vector(9 downto 0);
		ga_ls_order_8_o : out std_logic_vector(9 downto 0);
		ga_ls_order_9_o : out std_logic_vector(9 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ga is
	signal num_loads : std_logic_vector(3 downto 0);
	signal num_stores : std_logic_vector(3 downto 0);
	signal loads_sub : std_logic_vector(3 downto 0);
	signal stores_sub : std_logic_vector(3 downto 0);
	signal empty_loads : std_logic_vector(3 downto 0);
	signal empty_stores : std_logic_vector(3 downto 0);
	signal group_init_ready_0 : std_logic;
	signal group_init_hs_0 : std_logic;
	signal ga_ls_order_rom_0 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_1 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_2 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_3 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_4 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_5 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_6 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_7 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_8 : std_logic_vector(9 downto 0);
	signal ga_ls_order_rom_9 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_0 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_1 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_2 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_3 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_4 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_5 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_6 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_7 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_8 : std_logic_vector(9 downto 0);
	signal ga_ls_order_temp_9 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_0_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_1_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_2_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_3_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_4_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_5_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_6_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_7_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_8_0 : std_logic_vector(9 downto 0);
	signal TEMP_1_mux_9_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(3 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(3 downto 0);
	signal ldq_wen_unshifted_0 : std_logic;
	signal ldq_wen_unshifted_1 : std_logic;
	signal ldq_wen_unshifted_2 : std_logic;
	signal ldq_wen_unshifted_3 : std_logic;
	signal ldq_wen_unshifted_4 : std_logic;
	signal ldq_wen_unshifted_5 : std_logic;
	signal ldq_wen_unshifted_6 : std_logic;
	signal ldq_wen_unshifted_7 : std_logic;
	signal ldq_wen_unshifted_8 : std_logic;
	signal ldq_wen_unshifted_9 : std_logic;
	signal stq_wen_unshifted_0 : std_logic;
	signal stq_wen_unshifted_1 : std_logic;
	signal stq_wen_unshifted_2 : std_logic;
	signal stq_wen_unshifted_3 : std_logic;
	signal stq_wen_unshifted_4 : std_logic;
	signal stq_wen_unshifted_5 : std_logic;
	signal stq_wen_unshifted_6 : std_logic;
	signal stq_wen_unshifted_7 : std_logic;
	signal stq_wen_unshifted_8 : std_logic;
	signal stq_wen_unshifted_9 : std_logic;
	signal TEMP_4_res_0 : std_logic;
	signal TEMP_4_res_1 : std_logic;
	signal TEMP_4_res_2 : std_logic;
	signal TEMP_4_res_3 : std_logic;
	signal TEMP_4_res_4 : std_logic;
	signal TEMP_4_res_5 : std_logic;
	signal TEMP_4_res_6 : std_logic;
	signal TEMP_4_res_7 : std_logic;
	signal TEMP_4_res_8 : std_logic;
	signal TEMP_4_res_9 : std_logic;
	signal TEMP_5_res_0 : std_logic;
	signal TEMP_5_res_1 : std_logic;
	signal TEMP_5_res_2 : std_logic;
	signal TEMP_5_res_3 : std_logic;
	signal TEMP_5_res_4 : std_logic;
	signal TEMP_5_res_5 : std_logic;
	signal TEMP_5_res_6 : std_logic;
	signal TEMP_5_res_7 : std_logic;
	signal TEMP_5_res_8 : std_logic;
	signal TEMP_5_res_9 : std_logic;
	signal TEMP_6_res_0 : std_logic;
	signal TEMP_6_res_1 : std_logic;
	signal TEMP_6_res_2 : std_logic;
	signal TEMP_6_res_3 : std_logic;
	signal TEMP_6_res_4 : std_logic;
	signal TEMP_6_res_5 : std_logic;
	signal TEMP_6_res_6 : std_logic;
	signal TEMP_6_res_7 : std_logic;
	signal TEMP_6_res_8 : std_logic;
	signal TEMP_6_res_9 : std_logic;
	signal TEMP_7_res_0 : std_logic;
	signal TEMP_7_res_1 : std_logic;
	signal TEMP_7_res_2 : std_logic;
	signal TEMP_7_res_3 : std_logic;
	signal TEMP_7_res_4 : std_logic;
	signal TEMP_7_res_5 : std_logic;
	signal TEMP_7_res_6 : std_logic;
	signal TEMP_7_res_7 : std_logic;
	signal TEMP_7_res_8 : std_logic;
	signal TEMP_7_res_9 : std_logic;
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal TEMP_8_res_2 : std_logic;
	signal TEMP_8_res_3 : std_logic;
	signal TEMP_8_res_4 : std_logic;
	signal TEMP_8_res_5 : std_logic;
	signal TEMP_8_res_6 : std_logic;
	signal TEMP_8_res_7 : std_logic;
	signal TEMP_8_res_8 : std_logic;
	signal TEMP_8_res_9 : std_logic;
	signal TEMP_9_res_0 : std_logic;
	signal TEMP_9_res_1 : std_logic;
	signal TEMP_9_res_2 : std_logic;
	signal TEMP_9_res_3 : std_logic;
	signal TEMP_9_res_4 : std_logic;
	signal TEMP_9_res_5 : std_logic;
	signal TEMP_9_res_6 : std_logic;
	signal TEMP_9_res_7 : std_logic;
	signal TEMP_9_res_8 : std_logic;
	signal TEMP_9_res_9 : std_logic;
	signal TEMP_10_res : std_logic_vector(9 downto 0);
	signal TEMP_11_res : std_logic_vector(9 downto 0);
	signal TEMP_12_res : std_logic_vector(9 downto 0);
	signal TEMP_13_res : std_logic_vector(9 downto 0);
	signal TEMP_14_res : std_logic_vector(9 downto 0);
	signal TEMP_15_res : std_logic_vector(9 downto 0);
	signal TEMP_16_res : std_logic_vector(9 downto 0);
	signal TEMP_17_res : std_logic_vector(9 downto 0);
	signal TEMP_18_res : std_logic_vector(9 downto 0);
	signal TEMP_19_res : std_logic_vector(9 downto 0);
	signal TEMP_20_res : std_logic_vector(9 downto 0);
	signal TEMP_21_res : std_logic_vector(9 downto 0);
	signal TEMP_22_res : std_logic_vector(9 downto 0);
	signal TEMP_23_res : std_logic_vector(9 downto 0);
	signal TEMP_24_res : std_logic_vector(9 downto 0);
	signal TEMP_25_res : std_logic_vector(9 downto 0);
	signal TEMP_26_res : std_logic_vector(9 downto 0);
	signal TEMP_27_res : std_logic_vector(9 downto 0);
	signal TEMP_28_res : std_logic_vector(9 downto 0);
	signal TEMP_29_res : std_logic_vector(9 downto 0);
	signal TEMP_30_res : std_logic_vector(9 downto 0);
	signal TEMP_31_res : std_logic_vector(9 downto 0);
	signal TEMP_32_res : std_logic_vector(9 downto 0);
	signal TEMP_33_res : std_logic_vector(9 downto 0);
	signal TEMP_34_res : std_logic_vector(9 downto 0);
	signal TEMP_35_res : std_logic_vector(9 downto 0);
	signal TEMP_36_res : std_logic_vector(9 downto 0);
	signal TEMP_37_res : std_logic_vector(9 downto 0);
	signal TEMP_38_res : std_logic_vector(9 downto 0);
	signal TEMP_39_res : std_logic_vector(9 downto 0);
	signal TEMP_40_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_4 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_5 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_6 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_7 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_8 : std_logic_vector(9 downto 0);
	signal TEMP_40_res_9 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_4 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_5 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_6 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_7 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_8 : std_logic_vector(9 downto 0);
	signal TEMP_41_res_9 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_4 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_5 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_6 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_7 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_8 : std_logic_vector(9 downto 0);
	signal TEMP_42_res_9 : std_logic_vector(9 downto 0);
begin
	-- WrapSub Begin
	-- WrapSub(loads_sub, ldq_head, ldq_tail, 10)
	loads_sub <= std_logic_vector(unsigned(ldq_head_i) - unsigned(ldq_tail_i)) when ldq_head_i >= ldq_tail_i else
		std_logic_vector(10 - unsigned(ldq_tail_i) + unsigned(ldq_head_i));
	-- WrapAdd End

	-- WrapSub Begin
	-- WrapSub(stores_sub, stq_head, stq_tail, 10)
	stores_sub <= std_logic_vector(unsigned(stq_head_i) - unsigned(stq_tail_i)) when stq_head_i >= stq_tail_i else
		std_logic_vector(10 - unsigned(stq_tail_i) + unsigned(stq_head_i));
	-- WrapAdd End

	empty_loads <= "1010" when ldq_empty_i else loads_sub;
	empty_stores <= "1010" when stq_empty_i else stores_sub;
	group_init_ready_0 <= '1' when ( empty_loads >= "0001" ) and ( empty_stores >= "0001" ) else '0';
	group_init_ready_0_o <= group_init_ready_0;
	group_init_hs_0 <= group_init_ready_0 and group_init_valid_0_i;
	-- Mux1H For Rom Begin
	-- Mux1H(ga_ls_order_rom, group_init_hs)
	-- Loop 0
	TEMP_1_mux_0_0 <= "0000000000";
	ga_ls_order_rom_0 <= TEMP_1_mux_0_0;
	-- Loop 1
	TEMP_1_mux_1_0 <= "0000000000";
	ga_ls_order_rom_1 <= TEMP_1_mux_1_0;
	-- Loop 2
	TEMP_1_mux_2_0 <= "0000000000";
	ga_ls_order_rom_2 <= TEMP_1_mux_2_0;
	-- Loop 3
	TEMP_1_mux_3_0 <= "0000000000";
	ga_ls_order_rom_3 <= TEMP_1_mux_3_0;
	-- Loop 4
	TEMP_1_mux_4_0 <= "0000000000";
	ga_ls_order_rom_4 <= TEMP_1_mux_4_0;
	-- Loop 5
	TEMP_1_mux_5_0 <= "0000000000";
	ga_ls_order_rom_5 <= TEMP_1_mux_5_0;
	-- Loop 6
	TEMP_1_mux_6_0 <= "0000000000";
	ga_ls_order_rom_6 <= TEMP_1_mux_6_0;
	-- Loop 7
	TEMP_1_mux_7_0 <= "0000000000";
	ga_ls_order_rom_7 <= TEMP_1_mux_7_0;
	-- Loop 8
	TEMP_1_mux_8_0 <= "0000000000";
	ga_ls_order_rom_8 <= TEMP_1_mux_8_0;
	-- Loop 9
	TEMP_1_mux_9_0 <= "0000000000";
	ga_ls_order_rom_9 <= TEMP_1_mux_9_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_loads, group_init_hs)
	TEMP_2_mux_0 <= "0001" when group_init_hs_0 else "0000";
	num_loads <= TEMP_2_mux_0;
	-- Mux1H For Rom End

	-- Mux1H For Rom Begin
	-- Mux1H(num_stores, group_init_hs)
	TEMP_3_mux_0 <= "0001" when group_init_hs_0 else "0000";
	num_stores <= TEMP_3_mux_0;
	-- Mux1H For Rom End

	num_loads_o <= num_loads;
	num_stores_o <= num_stores;
	ldq_wen_unshifted_0 <= '1' when num_loads > "0000" else '0';
	ldq_wen_unshifted_1 <= '1' when num_loads > "0001" else '0';
	ldq_wen_unshifted_2 <= '1' when num_loads > "0010" else '0';
	ldq_wen_unshifted_3 <= '1' when num_loads > "0011" else '0';
	ldq_wen_unshifted_4 <= '1' when num_loads > "0100" else '0';
	ldq_wen_unshifted_5 <= '1' when num_loads > "0101" else '0';
	ldq_wen_unshifted_6 <= '1' when num_loads > "0110" else '0';
	ldq_wen_unshifted_7 <= '1' when num_loads > "0111" else '0';
	ldq_wen_unshifted_8 <= '1' when num_loads > "1000" else '0';
	ldq_wen_unshifted_9 <= '1' when num_loads > "1001" else '0';
	stq_wen_unshifted_0 <= '1' when num_stores > "0000" else '0';
	stq_wen_unshifted_1 <= '1' when num_stores > "0001" else '0';
	stq_wen_unshifted_2 <= '1' when num_stores > "0010" else '0';
	stq_wen_unshifted_3 <= '1' when num_stores > "0011" else '0';
	stq_wen_unshifted_4 <= '1' when num_stores > "0100" else '0';
	stq_wen_unshifted_5 <= '1' when num_stores > "0101" else '0';
	stq_wen_unshifted_6 <= '1' when num_stores > "0110" else '0';
	stq_wen_unshifted_7 <= '1' when num_stores > "0111" else '0';
	stq_wen_unshifted_8 <= '1' when num_stores > "1000" else '0';
	stq_wen_unshifted_9 <= '1' when num_stores > "1001" else '0';
	-- Shifter Begin
	-- CyclicLeftShift(ldq_wen, ldq_wen_unshifted, ldq_tail)
	TEMP_4_res_0 <= ldq_wen_unshifted_2 when ldq_tail_i(3) else ldq_wen_unshifted_0;
	TEMP_4_res_1 <= ldq_wen_unshifted_3 when ldq_tail_i(3) else ldq_wen_unshifted_1;
	TEMP_4_res_2 <= ldq_wen_unshifted_4 when ldq_tail_i(3) else ldq_wen_unshifted_2;
	TEMP_4_res_3 <= ldq_wen_unshifted_5 when ldq_tail_i(3) else ldq_wen_unshifted_3;
	TEMP_4_res_4 <= ldq_wen_unshifted_6 when ldq_tail_i(3) else ldq_wen_unshifted_4;
	TEMP_4_res_5 <= ldq_wen_unshifted_7 when ldq_tail_i(3) else ldq_wen_unshifted_5;
	TEMP_4_res_6 <= ldq_wen_unshifted_8 when ldq_tail_i(3) else ldq_wen_unshifted_6;
	TEMP_4_res_7 <= ldq_wen_unshifted_9 when ldq_tail_i(3) else ldq_wen_unshifted_7;
	TEMP_4_res_8 <= ldq_wen_unshifted_0 when ldq_tail_i(3) else ldq_wen_unshifted_8;
	TEMP_4_res_9 <= ldq_wen_unshifted_1 when ldq_tail_i(3) else ldq_wen_unshifted_9;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_6 when ldq_tail_i(2) else TEMP_4_res_0;
	TEMP_5_res_1 <= TEMP_4_res_7 when ldq_tail_i(2) else TEMP_4_res_1;
	TEMP_5_res_2 <= TEMP_4_res_8 when ldq_tail_i(2) else TEMP_4_res_2;
	TEMP_5_res_3 <= TEMP_4_res_9 when ldq_tail_i(2) else TEMP_4_res_3;
	TEMP_5_res_4 <= TEMP_4_res_0 when ldq_tail_i(2) else TEMP_4_res_4;
	TEMP_5_res_5 <= TEMP_4_res_1 when ldq_tail_i(2) else TEMP_4_res_5;
	TEMP_5_res_6 <= TEMP_4_res_2 when ldq_tail_i(2) else TEMP_4_res_6;
	TEMP_5_res_7 <= TEMP_4_res_3 when ldq_tail_i(2) else TEMP_4_res_7;
	TEMP_5_res_8 <= TEMP_4_res_4 when ldq_tail_i(2) else TEMP_4_res_8;
	TEMP_5_res_9 <= TEMP_4_res_5 when ldq_tail_i(2) else TEMP_4_res_9;
	-- Layer End
	TEMP_6_res_0 <= TEMP_5_res_8 when ldq_tail_i(1) else TEMP_5_res_0;
	TEMP_6_res_1 <= TEMP_5_res_9 when ldq_tail_i(1) else TEMP_5_res_1;
	TEMP_6_res_2 <= TEMP_5_res_0 when ldq_tail_i(1) else TEMP_5_res_2;
	TEMP_6_res_3 <= TEMP_5_res_1 when ldq_tail_i(1) else TEMP_5_res_3;
	TEMP_6_res_4 <= TEMP_5_res_2 when ldq_tail_i(1) else TEMP_5_res_4;
	TEMP_6_res_5 <= TEMP_5_res_3 when ldq_tail_i(1) else TEMP_5_res_5;
	TEMP_6_res_6 <= TEMP_5_res_4 when ldq_tail_i(1) else TEMP_5_res_6;
	TEMP_6_res_7 <= TEMP_5_res_5 when ldq_tail_i(1) else TEMP_5_res_7;
	TEMP_6_res_8 <= TEMP_5_res_6 when ldq_tail_i(1) else TEMP_5_res_8;
	TEMP_6_res_9 <= TEMP_5_res_7 when ldq_tail_i(1) else TEMP_5_res_9;
	-- Layer End
	ldq_wen_0_o <= TEMP_6_res_9 when ldq_tail_i(0) else TEMP_6_res_0;
	ldq_wen_1_o <= TEMP_6_res_0 when ldq_tail_i(0) else TEMP_6_res_1;
	ldq_wen_2_o <= TEMP_6_res_1 when ldq_tail_i(0) else TEMP_6_res_2;
	ldq_wen_3_o <= TEMP_6_res_2 when ldq_tail_i(0) else TEMP_6_res_3;
	ldq_wen_4_o <= TEMP_6_res_3 when ldq_tail_i(0) else TEMP_6_res_4;
	ldq_wen_5_o <= TEMP_6_res_4 when ldq_tail_i(0) else TEMP_6_res_5;
	ldq_wen_6_o <= TEMP_6_res_5 when ldq_tail_i(0) else TEMP_6_res_6;
	ldq_wen_7_o <= TEMP_6_res_6 when ldq_tail_i(0) else TEMP_6_res_7;
	ldq_wen_8_o <= TEMP_6_res_7 when ldq_tail_i(0) else TEMP_6_res_8;
	ldq_wen_9_o <= TEMP_6_res_8 when ldq_tail_i(0) else TEMP_6_res_9;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(stq_wen, stq_wen_unshifted, stq_tail)
	TEMP_7_res_0 <= stq_wen_unshifted_2 when stq_tail_i(3) else stq_wen_unshifted_0;
	TEMP_7_res_1 <= stq_wen_unshifted_3 when stq_tail_i(3) else stq_wen_unshifted_1;
	TEMP_7_res_2 <= stq_wen_unshifted_4 when stq_tail_i(3) else stq_wen_unshifted_2;
	TEMP_7_res_3 <= stq_wen_unshifted_5 when stq_tail_i(3) else stq_wen_unshifted_3;
	TEMP_7_res_4 <= stq_wen_unshifted_6 when stq_tail_i(3) else stq_wen_unshifted_4;
	TEMP_7_res_5 <= stq_wen_unshifted_7 when stq_tail_i(3) else stq_wen_unshifted_5;
	TEMP_7_res_6 <= stq_wen_unshifted_8 when stq_tail_i(3) else stq_wen_unshifted_6;
	TEMP_7_res_7 <= stq_wen_unshifted_9 when stq_tail_i(3) else stq_wen_unshifted_7;
	TEMP_7_res_8 <= stq_wen_unshifted_0 when stq_tail_i(3) else stq_wen_unshifted_8;
	TEMP_7_res_9 <= stq_wen_unshifted_1 when stq_tail_i(3) else stq_wen_unshifted_9;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_6 when stq_tail_i(2) else TEMP_7_res_0;
	TEMP_8_res_1 <= TEMP_7_res_7 when stq_tail_i(2) else TEMP_7_res_1;
	TEMP_8_res_2 <= TEMP_7_res_8 when stq_tail_i(2) else TEMP_7_res_2;
	TEMP_8_res_3 <= TEMP_7_res_9 when stq_tail_i(2) else TEMP_7_res_3;
	TEMP_8_res_4 <= TEMP_7_res_0 when stq_tail_i(2) else TEMP_7_res_4;
	TEMP_8_res_5 <= TEMP_7_res_1 when stq_tail_i(2) else TEMP_7_res_5;
	TEMP_8_res_6 <= TEMP_7_res_2 when stq_tail_i(2) else TEMP_7_res_6;
	TEMP_8_res_7 <= TEMP_7_res_3 when stq_tail_i(2) else TEMP_7_res_7;
	TEMP_8_res_8 <= TEMP_7_res_4 when stq_tail_i(2) else TEMP_7_res_8;
	TEMP_8_res_9 <= TEMP_7_res_5 when stq_tail_i(2) else TEMP_7_res_9;
	-- Layer End
	TEMP_9_res_0 <= TEMP_8_res_8 when stq_tail_i(1) else TEMP_8_res_0;
	TEMP_9_res_1 <= TEMP_8_res_9 when stq_tail_i(1) else TEMP_8_res_1;
	TEMP_9_res_2 <= TEMP_8_res_0 when stq_tail_i(1) else TEMP_8_res_2;
	TEMP_9_res_3 <= TEMP_8_res_1 when stq_tail_i(1) else TEMP_8_res_3;
	TEMP_9_res_4 <= TEMP_8_res_2 when stq_tail_i(1) else TEMP_8_res_4;
	TEMP_9_res_5 <= TEMP_8_res_3 when stq_tail_i(1) else TEMP_8_res_5;
	TEMP_9_res_6 <= TEMP_8_res_4 when stq_tail_i(1) else TEMP_8_res_6;
	TEMP_9_res_7 <= TEMP_8_res_5 when stq_tail_i(1) else TEMP_8_res_7;
	TEMP_9_res_8 <= TEMP_8_res_6 when stq_tail_i(1) else TEMP_8_res_8;
	TEMP_9_res_9 <= TEMP_8_res_7 when stq_tail_i(1) else TEMP_8_res_9;
	-- Layer End
	stq_wen_0_o <= TEMP_9_res_9 when stq_tail_i(0) else TEMP_9_res_0;
	stq_wen_1_o <= TEMP_9_res_0 when stq_tail_i(0) else TEMP_9_res_1;
	stq_wen_2_o <= TEMP_9_res_1 when stq_tail_i(0) else TEMP_9_res_2;
	stq_wen_3_o <= TEMP_9_res_2 when stq_tail_i(0) else TEMP_9_res_3;
	stq_wen_4_o <= TEMP_9_res_3 when stq_tail_i(0) else TEMP_9_res_4;
	stq_wen_5_o <= TEMP_9_res_4 when stq_tail_i(0) else TEMP_9_res_5;
	stq_wen_6_o <= TEMP_9_res_5 when stq_tail_i(0) else TEMP_9_res_6;
	stq_wen_7_o <= TEMP_9_res_6 when stq_tail_i(0) else TEMP_9_res_7;
	stq_wen_8_o <= TEMP_9_res_7 when stq_tail_i(0) else TEMP_9_res_8;
	stq_wen_9_o <= TEMP_9_res_8 when stq_tail_i(0) else TEMP_9_res_9;
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_0, ga_ls_order_rom_0, stq_tail)
	TEMP_10_res(0) <= ga_ls_order_rom_0(2) when stq_tail_i(3) else ga_ls_order_rom_0(0);
	TEMP_10_res(1) <= ga_ls_order_rom_0(3) when stq_tail_i(3) else ga_ls_order_rom_0(1);
	TEMP_10_res(2) <= ga_ls_order_rom_0(4) when stq_tail_i(3) else ga_ls_order_rom_0(2);
	TEMP_10_res(3) <= ga_ls_order_rom_0(5) when stq_tail_i(3) else ga_ls_order_rom_0(3);
	TEMP_10_res(4) <= ga_ls_order_rom_0(6) when stq_tail_i(3) else ga_ls_order_rom_0(4);
	TEMP_10_res(5) <= ga_ls_order_rom_0(7) when stq_tail_i(3) else ga_ls_order_rom_0(5);
	TEMP_10_res(6) <= ga_ls_order_rom_0(8) when stq_tail_i(3) else ga_ls_order_rom_0(6);
	TEMP_10_res(7) <= ga_ls_order_rom_0(9) when stq_tail_i(3) else ga_ls_order_rom_0(7);
	TEMP_10_res(8) <= ga_ls_order_rom_0(0) when stq_tail_i(3) else ga_ls_order_rom_0(8);
	TEMP_10_res(9) <= ga_ls_order_rom_0(1) when stq_tail_i(3) else ga_ls_order_rom_0(9);
	-- Layer End
	TEMP_11_res(0) <= TEMP_10_res(6) when stq_tail_i(2) else TEMP_10_res(0);
	TEMP_11_res(1) <= TEMP_10_res(7) when stq_tail_i(2) else TEMP_10_res(1);
	TEMP_11_res(2) <= TEMP_10_res(8) when stq_tail_i(2) else TEMP_10_res(2);
	TEMP_11_res(3) <= TEMP_10_res(9) when stq_tail_i(2) else TEMP_10_res(3);
	TEMP_11_res(4) <= TEMP_10_res(0) when stq_tail_i(2) else TEMP_10_res(4);
	TEMP_11_res(5) <= TEMP_10_res(1) when stq_tail_i(2) else TEMP_10_res(5);
	TEMP_11_res(6) <= TEMP_10_res(2) when stq_tail_i(2) else TEMP_10_res(6);
	TEMP_11_res(7) <= TEMP_10_res(3) when stq_tail_i(2) else TEMP_10_res(7);
	TEMP_11_res(8) <= TEMP_10_res(4) when stq_tail_i(2) else TEMP_10_res(8);
	TEMP_11_res(9) <= TEMP_10_res(5) when stq_tail_i(2) else TEMP_10_res(9);
	-- Layer End
	TEMP_12_res(0) <= TEMP_11_res(8) when stq_tail_i(1) else TEMP_11_res(0);
	TEMP_12_res(1) <= TEMP_11_res(9) when stq_tail_i(1) else TEMP_11_res(1);
	TEMP_12_res(2) <= TEMP_11_res(0) when stq_tail_i(1) else TEMP_11_res(2);
	TEMP_12_res(3) <= TEMP_11_res(1) when stq_tail_i(1) else TEMP_11_res(3);
	TEMP_12_res(4) <= TEMP_11_res(2) when stq_tail_i(1) else TEMP_11_res(4);
	TEMP_12_res(5) <= TEMP_11_res(3) when stq_tail_i(1) else TEMP_11_res(5);
	TEMP_12_res(6) <= TEMP_11_res(4) when stq_tail_i(1) else TEMP_11_res(6);
	TEMP_12_res(7) <= TEMP_11_res(5) when stq_tail_i(1) else TEMP_11_res(7);
	TEMP_12_res(8) <= TEMP_11_res(6) when stq_tail_i(1) else TEMP_11_res(8);
	TEMP_12_res(9) <= TEMP_11_res(7) when stq_tail_i(1) else TEMP_11_res(9);
	-- Layer End
	ga_ls_order_temp_0(0) <= TEMP_12_res(9) when stq_tail_i(0) else TEMP_12_res(0);
	ga_ls_order_temp_0(1) <= TEMP_12_res(0) when stq_tail_i(0) else TEMP_12_res(1);
	ga_ls_order_temp_0(2) <= TEMP_12_res(1) when stq_tail_i(0) else TEMP_12_res(2);
	ga_ls_order_temp_0(3) <= TEMP_12_res(2) when stq_tail_i(0) else TEMP_12_res(3);
	ga_ls_order_temp_0(4) <= TEMP_12_res(3) when stq_tail_i(0) else TEMP_12_res(4);
	ga_ls_order_temp_0(5) <= TEMP_12_res(4) when stq_tail_i(0) else TEMP_12_res(5);
	ga_ls_order_temp_0(6) <= TEMP_12_res(5) when stq_tail_i(0) else TEMP_12_res(6);
	ga_ls_order_temp_0(7) <= TEMP_12_res(6) when stq_tail_i(0) else TEMP_12_res(7);
	ga_ls_order_temp_0(8) <= TEMP_12_res(7) when stq_tail_i(0) else TEMP_12_res(8);
	ga_ls_order_temp_0(9) <= TEMP_12_res(8) when stq_tail_i(0) else TEMP_12_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_1, ga_ls_order_rom_1, stq_tail)
	TEMP_13_res(0) <= ga_ls_order_rom_1(2) when stq_tail_i(3) else ga_ls_order_rom_1(0);
	TEMP_13_res(1) <= ga_ls_order_rom_1(3) when stq_tail_i(3) else ga_ls_order_rom_1(1);
	TEMP_13_res(2) <= ga_ls_order_rom_1(4) when stq_tail_i(3) else ga_ls_order_rom_1(2);
	TEMP_13_res(3) <= ga_ls_order_rom_1(5) when stq_tail_i(3) else ga_ls_order_rom_1(3);
	TEMP_13_res(4) <= ga_ls_order_rom_1(6) when stq_tail_i(3) else ga_ls_order_rom_1(4);
	TEMP_13_res(5) <= ga_ls_order_rom_1(7) when stq_tail_i(3) else ga_ls_order_rom_1(5);
	TEMP_13_res(6) <= ga_ls_order_rom_1(8) when stq_tail_i(3) else ga_ls_order_rom_1(6);
	TEMP_13_res(7) <= ga_ls_order_rom_1(9) when stq_tail_i(3) else ga_ls_order_rom_1(7);
	TEMP_13_res(8) <= ga_ls_order_rom_1(0) when stq_tail_i(3) else ga_ls_order_rom_1(8);
	TEMP_13_res(9) <= ga_ls_order_rom_1(1) when stq_tail_i(3) else ga_ls_order_rom_1(9);
	-- Layer End
	TEMP_14_res(0) <= TEMP_13_res(6) when stq_tail_i(2) else TEMP_13_res(0);
	TEMP_14_res(1) <= TEMP_13_res(7) when stq_tail_i(2) else TEMP_13_res(1);
	TEMP_14_res(2) <= TEMP_13_res(8) when stq_tail_i(2) else TEMP_13_res(2);
	TEMP_14_res(3) <= TEMP_13_res(9) when stq_tail_i(2) else TEMP_13_res(3);
	TEMP_14_res(4) <= TEMP_13_res(0) when stq_tail_i(2) else TEMP_13_res(4);
	TEMP_14_res(5) <= TEMP_13_res(1) when stq_tail_i(2) else TEMP_13_res(5);
	TEMP_14_res(6) <= TEMP_13_res(2) when stq_tail_i(2) else TEMP_13_res(6);
	TEMP_14_res(7) <= TEMP_13_res(3) when stq_tail_i(2) else TEMP_13_res(7);
	TEMP_14_res(8) <= TEMP_13_res(4) when stq_tail_i(2) else TEMP_13_res(8);
	TEMP_14_res(9) <= TEMP_13_res(5) when stq_tail_i(2) else TEMP_13_res(9);
	-- Layer End
	TEMP_15_res(0) <= TEMP_14_res(8) when stq_tail_i(1) else TEMP_14_res(0);
	TEMP_15_res(1) <= TEMP_14_res(9) when stq_tail_i(1) else TEMP_14_res(1);
	TEMP_15_res(2) <= TEMP_14_res(0) when stq_tail_i(1) else TEMP_14_res(2);
	TEMP_15_res(3) <= TEMP_14_res(1) when stq_tail_i(1) else TEMP_14_res(3);
	TEMP_15_res(4) <= TEMP_14_res(2) when stq_tail_i(1) else TEMP_14_res(4);
	TEMP_15_res(5) <= TEMP_14_res(3) when stq_tail_i(1) else TEMP_14_res(5);
	TEMP_15_res(6) <= TEMP_14_res(4) when stq_tail_i(1) else TEMP_14_res(6);
	TEMP_15_res(7) <= TEMP_14_res(5) when stq_tail_i(1) else TEMP_14_res(7);
	TEMP_15_res(8) <= TEMP_14_res(6) when stq_tail_i(1) else TEMP_14_res(8);
	TEMP_15_res(9) <= TEMP_14_res(7) when stq_tail_i(1) else TEMP_14_res(9);
	-- Layer End
	ga_ls_order_temp_1(0) <= TEMP_15_res(9) when stq_tail_i(0) else TEMP_15_res(0);
	ga_ls_order_temp_1(1) <= TEMP_15_res(0) when stq_tail_i(0) else TEMP_15_res(1);
	ga_ls_order_temp_1(2) <= TEMP_15_res(1) when stq_tail_i(0) else TEMP_15_res(2);
	ga_ls_order_temp_1(3) <= TEMP_15_res(2) when stq_tail_i(0) else TEMP_15_res(3);
	ga_ls_order_temp_1(4) <= TEMP_15_res(3) when stq_tail_i(0) else TEMP_15_res(4);
	ga_ls_order_temp_1(5) <= TEMP_15_res(4) when stq_tail_i(0) else TEMP_15_res(5);
	ga_ls_order_temp_1(6) <= TEMP_15_res(5) when stq_tail_i(0) else TEMP_15_res(6);
	ga_ls_order_temp_1(7) <= TEMP_15_res(6) when stq_tail_i(0) else TEMP_15_res(7);
	ga_ls_order_temp_1(8) <= TEMP_15_res(7) when stq_tail_i(0) else TEMP_15_res(8);
	ga_ls_order_temp_1(9) <= TEMP_15_res(8) when stq_tail_i(0) else TEMP_15_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_2, ga_ls_order_rom_2, stq_tail)
	TEMP_16_res(0) <= ga_ls_order_rom_2(2) when stq_tail_i(3) else ga_ls_order_rom_2(0);
	TEMP_16_res(1) <= ga_ls_order_rom_2(3) when stq_tail_i(3) else ga_ls_order_rom_2(1);
	TEMP_16_res(2) <= ga_ls_order_rom_2(4) when stq_tail_i(3) else ga_ls_order_rom_2(2);
	TEMP_16_res(3) <= ga_ls_order_rom_2(5) when stq_tail_i(3) else ga_ls_order_rom_2(3);
	TEMP_16_res(4) <= ga_ls_order_rom_2(6) when stq_tail_i(3) else ga_ls_order_rom_2(4);
	TEMP_16_res(5) <= ga_ls_order_rom_2(7) when stq_tail_i(3) else ga_ls_order_rom_2(5);
	TEMP_16_res(6) <= ga_ls_order_rom_2(8) when stq_tail_i(3) else ga_ls_order_rom_2(6);
	TEMP_16_res(7) <= ga_ls_order_rom_2(9) when stq_tail_i(3) else ga_ls_order_rom_2(7);
	TEMP_16_res(8) <= ga_ls_order_rom_2(0) when stq_tail_i(3) else ga_ls_order_rom_2(8);
	TEMP_16_res(9) <= ga_ls_order_rom_2(1) when stq_tail_i(3) else ga_ls_order_rom_2(9);
	-- Layer End
	TEMP_17_res(0) <= TEMP_16_res(6) when stq_tail_i(2) else TEMP_16_res(0);
	TEMP_17_res(1) <= TEMP_16_res(7) when stq_tail_i(2) else TEMP_16_res(1);
	TEMP_17_res(2) <= TEMP_16_res(8) when stq_tail_i(2) else TEMP_16_res(2);
	TEMP_17_res(3) <= TEMP_16_res(9) when stq_tail_i(2) else TEMP_16_res(3);
	TEMP_17_res(4) <= TEMP_16_res(0) when stq_tail_i(2) else TEMP_16_res(4);
	TEMP_17_res(5) <= TEMP_16_res(1) when stq_tail_i(2) else TEMP_16_res(5);
	TEMP_17_res(6) <= TEMP_16_res(2) when stq_tail_i(2) else TEMP_16_res(6);
	TEMP_17_res(7) <= TEMP_16_res(3) when stq_tail_i(2) else TEMP_16_res(7);
	TEMP_17_res(8) <= TEMP_16_res(4) when stq_tail_i(2) else TEMP_16_res(8);
	TEMP_17_res(9) <= TEMP_16_res(5) when stq_tail_i(2) else TEMP_16_res(9);
	-- Layer End
	TEMP_18_res(0) <= TEMP_17_res(8) when stq_tail_i(1) else TEMP_17_res(0);
	TEMP_18_res(1) <= TEMP_17_res(9) when stq_tail_i(1) else TEMP_17_res(1);
	TEMP_18_res(2) <= TEMP_17_res(0) when stq_tail_i(1) else TEMP_17_res(2);
	TEMP_18_res(3) <= TEMP_17_res(1) when stq_tail_i(1) else TEMP_17_res(3);
	TEMP_18_res(4) <= TEMP_17_res(2) when stq_tail_i(1) else TEMP_17_res(4);
	TEMP_18_res(5) <= TEMP_17_res(3) when stq_tail_i(1) else TEMP_17_res(5);
	TEMP_18_res(6) <= TEMP_17_res(4) when stq_tail_i(1) else TEMP_17_res(6);
	TEMP_18_res(7) <= TEMP_17_res(5) when stq_tail_i(1) else TEMP_17_res(7);
	TEMP_18_res(8) <= TEMP_17_res(6) when stq_tail_i(1) else TEMP_17_res(8);
	TEMP_18_res(9) <= TEMP_17_res(7) when stq_tail_i(1) else TEMP_17_res(9);
	-- Layer End
	ga_ls_order_temp_2(0) <= TEMP_18_res(9) when stq_tail_i(0) else TEMP_18_res(0);
	ga_ls_order_temp_2(1) <= TEMP_18_res(0) when stq_tail_i(0) else TEMP_18_res(1);
	ga_ls_order_temp_2(2) <= TEMP_18_res(1) when stq_tail_i(0) else TEMP_18_res(2);
	ga_ls_order_temp_2(3) <= TEMP_18_res(2) when stq_tail_i(0) else TEMP_18_res(3);
	ga_ls_order_temp_2(4) <= TEMP_18_res(3) when stq_tail_i(0) else TEMP_18_res(4);
	ga_ls_order_temp_2(5) <= TEMP_18_res(4) when stq_tail_i(0) else TEMP_18_res(5);
	ga_ls_order_temp_2(6) <= TEMP_18_res(5) when stq_tail_i(0) else TEMP_18_res(6);
	ga_ls_order_temp_2(7) <= TEMP_18_res(6) when stq_tail_i(0) else TEMP_18_res(7);
	ga_ls_order_temp_2(8) <= TEMP_18_res(7) when stq_tail_i(0) else TEMP_18_res(8);
	ga_ls_order_temp_2(9) <= TEMP_18_res(8) when stq_tail_i(0) else TEMP_18_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_3, ga_ls_order_rom_3, stq_tail)
	TEMP_19_res(0) <= ga_ls_order_rom_3(2) when stq_tail_i(3) else ga_ls_order_rom_3(0);
	TEMP_19_res(1) <= ga_ls_order_rom_3(3) when stq_tail_i(3) else ga_ls_order_rom_3(1);
	TEMP_19_res(2) <= ga_ls_order_rom_3(4) when stq_tail_i(3) else ga_ls_order_rom_3(2);
	TEMP_19_res(3) <= ga_ls_order_rom_3(5) when stq_tail_i(3) else ga_ls_order_rom_3(3);
	TEMP_19_res(4) <= ga_ls_order_rom_3(6) when stq_tail_i(3) else ga_ls_order_rom_3(4);
	TEMP_19_res(5) <= ga_ls_order_rom_3(7) when stq_tail_i(3) else ga_ls_order_rom_3(5);
	TEMP_19_res(6) <= ga_ls_order_rom_3(8) when stq_tail_i(3) else ga_ls_order_rom_3(6);
	TEMP_19_res(7) <= ga_ls_order_rom_3(9) when stq_tail_i(3) else ga_ls_order_rom_3(7);
	TEMP_19_res(8) <= ga_ls_order_rom_3(0) when stq_tail_i(3) else ga_ls_order_rom_3(8);
	TEMP_19_res(9) <= ga_ls_order_rom_3(1) when stq_tail_i(3) else ga_ls_order_rom_3(9);
	-- Layer End
	TEMP_20_res(0) <= TEMP_19_res(6) when stq_tail_i(2) else TEMP_19_res(0);
	TEMP_20_res(1) <= TEMP_19_res(7) when stq_tail_i(2) else TEMP_19_res(1);
	TEMP_20_res(2) <= TEMP_19_res(8) when stq_tail_i(2) else TEMP_19_res(2);
	TEMP_20_res(3) <= TEMP_19_res(9) when stq_tail_i(2) else TEMP_19_res(3);
	TEMP_20_res(4) <= TEMP_19_res(0) when stq_tail_i(2) else TEMP_19_res(4);
	TEMP_20_res(5) <= TEMP_19_res(1) when stq_tail_i(2) else TEMP_19_res(5);
	TEMP_20_res(6) <= TEMP_19_res(2) when stq_tail_i(2) else TEMP_19_res(6);
	TEMP_20_res(7) <= TEMP_19_res(3) when stq_tail_i(2) else TEMP_19_res(7);
	TEMP_20_res(8) <= TEMP_19_res(4) when stq_tail_i(2) else TEMP_19_res(8);
	TEMP_20_res(9) <= TEMP_19_res(5) when stq_tail_i(2) else TEMP_19_res(9);
	-- Layer End
	TEMP_21_res(0) <= TEMP_20_res(8) when stq_tail_i(1) else TEMP_20_res(0);
	TEMP_21_res(1) <= TEMP_20_res(9) when stq_tail_i(1) else TEMP_20_res(1);
	TEMP_21_res(2) <= TEMP_20_res(0) when stq_tail_i(1) else TEMP_20_res(2);
	TEMP_21_res(3) <= TEMP_20_res(1) when stq_tail_i(1) else TEMP_20_res(3);
	TEMP_21_res(4) <= TEMP_20_res(2) when stq_tail_i(1) else TEMP_20_res(4);
	TEMP_21_res(5) <= TEMP_20_res(3) when stq_tail_i(1) else TEMP_20_res(5);
	TEMP_21_res(6) <= TEMP_20_res(4) when stq_tail_i(1) else TEMP_20_res(6);
	TEMP_21_res(7) <= TEMP_20_res(5) when stq_tail_i(1) else TEMP_20_res(7);
	TEMP_21_res(8) <= TEMP_20_res(6) when stq_tail_i(1) else TEMP_20_res(8);
	TEMP_21_res(9) <= TEMP_20_res(7) when stq_tail_i(1) else TEMP_20_res(9);
	-- Layer End
	ga_ls_order_temp_3(0) <= TEMP_21_res(9) when stq_tail_i(0) else TEMP_21_res(0);
	ga_ls_order_temp_3(1) <= TEMP_21_res(0) when stq_tail_i(0) else TEMP_21_res(1);
	ga_ls_order_temp_3(2) <= TEMP_21_res(1) when stq_tail_i(0) else TEMP_21_res(2);
	ga_ls_order_temp_3(3) <= TEMP_21_res(2) when stq_tail_i(0) else TEMP_21_res(3);
	ga_ls_order_temp_3(4) <= TEMP_21_res(3) when stq_tail_i(0) else TEMP_21_res(4);
	ga_ls_order_temp_3(5) <= TEMP_21_res(4) when stq_tail_i(0) else TEMP_21_res(5);
	ga_ls_order_temp_3(6) <= TEMP_21_res(5) when stq_tail_i(0) else TEMP_21_res(6);
	ga_ls_order_temp_3(7) <= TEMP_21_res(6) when stq_tail_i(0) else TEMP_21_res(7);
	ga_ls_order_temp_3(8) <= TEMP_21_res(7) when stq_tail_i(0) else TEMP_21_res(8);
	ga_ls_order_temp_3(9) <= TEMP_21_res(8) when stq_tail_i(0) else TEMP_21_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_4, ga_ls_order_rom_4, stq_tail)
	TEMP_22_res(0) <= ga_ls_order_rom_4(2) when stq_tail_i(3) else ga_ls_order_rom_4(0);
	TEMP_22_res(1) <= ga_ls_order_rom_4(3) when stq_tail_i(3) else ga_ls_order_rom_4(1);
	TEMP_22_res(2) <= ga_ls_order_rom_4(4) when stq_tail_i(3) else ga_ls_order_rom_4(2);
	TEMP_22_res(3) <= ga_ls_order_rom_4(5) when stq_tail_i(3) else ga_ls_order_rom_4(3);
	TEMP_22_res(4) <= ga_ls_order_rom_4(6) when stq_tail_i(3) else ga_ls_order_rom_4(4);
	TEMP_22_res(5) <= ga_ls_order_rom_4(7) when stq_tail_i(3) else ga_ls_order_rom_4(5);
	TEMP_22_res(6) <= ga_ls_order_rom_4(8) when stq_tail_i(3) else ga_ls_order_rom_4(6);
	TEMP_22_res(7) <= ga_ls_order_rom_4(9) when stq_tail_i(3) else ga_ls_order_rom_4(7);
	TEMP_22_res(8) <= ga_ls_order_rom_4(0) when stq_tail_i(3) else ga_ls_order_rom_4(8);
	TEMP_22_res(9) <= ga_ls_order_rom_4(1) when stq_tail_i(3) else ga_ls_order_rom_4(9);
	-- Layer End
	TEMP_23_res(0) <= TEMP_22_res(6) when stq_tail_i(2) else TEMP_22_res(0);
	TEMP_23_res(1) <= TEMP_22_res(7) when stq_tail_i(2) else TEMP_22_res(1);
	TEMP_23_res(2) <= TEMP_22_res(8) when stq_tail_i(2) else TEMP_22_res(2);
	TEMP_23_res(3) <= TEMP_22_res(9) when stq_tail_i(2) else TEMP_22_res(3);
	TEMP_23_res(4) <= TEMP_22_res(0) when stq_tail_i(2) else TEMP_22_res(4);
	TEMP_23_res(5) <= TEMP_22_res(1) when stq_tail_i(2) else TEMP_22_res(5);
	TEMP_23_res(6) <= TEMP_22_res(2) when stq_tail_i(2) else TEMP_22_res(6);
	TEMP_23_res(7) <= TEMP_22_res(3) when stq_tail_i(2) else TEMP_22_res(7);
	TEMP_23_res(8) <= TEMP_22_res(4) when stq_tail_i(2) else TEMP_22_res(8);
	TEMP_23_res(9) <= TEMP_22_res(5) when stq_tail_i(2) else TEMP_22_res(9);
	-- Layer End
	TEMP_24_res(0) <= TEMP_23_res(8) when stq_tail_i(1) else TEMP_23_res(0);
	TEMP_24_res(1) <= TEMP_23_res(9) when stq_tail_i(1) else TEMP_23_res(1);
	TEMP_24_res(2) <= TEMP_23_res(0) when stq_tail_i(1) else TEMP_23_res(2);
	TEMP_24_res(3) <= TEMP_23_res(1) when stq_tail_i(1) else TEMP_23_res(3);
	TEMP_24_res(4) <= TEMP_23_res(2) when stq_tail_i(1) else TEMP_23_res(4);
	TEMP_24_res(5) <= TEMP_23_res(3) when stq_tail_i(1) else TEMP_23_res(5);
	TEMP_24_res(6) <= TEMP_23_res(4) when stq_tail_i(1) else TEMP_23_res(6);
	TEMP_24_res(7) <= TEMP_23_res(5) when stq_tail_i(1) else TEMP_23_res(7);
	TEMP_24_res(8) <= TEMP_23_res(6) when stq_tail_i(1) else TEMP_23_res(8);
	TEMP_24_res(9) <= TEMP_23_res(7) when stq_tail_i(1) else TEMP_23_res(9);
	-- Layer End
	ga_ls_order_temp_4(0) <= TEMP_24_res(9) when stq_tail_i(0) else TEMP_24_res(0);
	ga_ls_order_temp_4(1) <= TEMP_24_res(0) when stq_tail_i(0) else TEMP_24_res(1);
	ga_ls_order_temp_4(2) <= TEMP_24_res(1) when stq_tail_i(0) else TEMP_24_res(2);
	ga_ls_order_temp_4(3) <= TEMP_24_res(2) when stq_tail_i(0) else TEMP_24_res(3);
	ga_ls_order_temp_4(4) <= TEMP_24_res(3) when stq_tail_i(0) else TEMP_24_res(4);
	ga_ls_order_temp_4(5) <= TEMP_24_res(4) when stq_tail_i(0) else TEMP_24_res(5);
	ga_ls_order_temp_4(6) <= TEMP_24_res(5) when stq_tail_i(0) else TEMP_24_res(6);
	ga_ls_order_temp_4(7) <= TEMP_24_res(6) when stq_tail_i(0) else TEMP_24_res(7);
	ga_ls_order_temp_4(8) <= TEMP_24_res(7) when stq_tail_i(0) else TEMP_24_res(8);
	ga_ls_order_temp_4(9) <= TEMP_24_res(8) when stq_tail_i(0) else TEMP_24_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_5, ga_ls_order_rom_5, stq_tail)
	TEMP_25_res(0) <= ga_ls_order_rom_5(2) when stq_tail_i(3) else ga_ls_order_rom_5(0);
	TEMP_25_res(1) <= ga_ls_order_rom_5(3) when stq_tail_i(3) else ga_ls_order_rom_5(1);
	TEMP_25_res(2) <= ga_ls_order_rom_5(4) when stq_tail_i(3) else ga_ls_order_rom_5(2);
	TEMP_25_res(3) <= ga_ls_order_rom_5(5) when stq_tail_i(3) else ga_ls_order_rom_5(3);
	TEMP_25_res(4) <= ga_ls_order_rom_5(6) when stq_tail_i(3) else ga_ls_order_rom_5(4);
	TEMP_25_res(5) <= ga_ls_order_rom_5(7) when stq_tail_i(3) else ga_ls_order_rom_5(5);
	TEMP_25_res(6) <= ga_ls_order_rom_5(8) when stq_tail_i(3) else ga_ls_order_rom_5(6);
	TEMP_25_res(7) <= ga_ls_order_rom_5(9) when stq_tail_i(3) else ga_ls_order_rom_5(7);
	TEMP_25_res(8) <= ga_ls_order_rom_5(0) when stq_tail_i(3) else ga_ls_order_rom_5(8);
	TEMP_25_res(9) <= ga_ls_order_rom_5(1) when stq_tail_i(3) else ga_ls_order_rom_5(9);
	-- Layer End
	TEMP_26_res(0) <= TEMP_25_res(6) when stq_tail_i(2) else TEMP_25_res(0);
	TEMP_26_res(1) <= TEMP_25_res(7) when stq_tail_i(2) else TEMP_25_res(1);
	TEMP_26_res(2) <= TEMP_25_res(8) when stq_tail_i(2) else TEMP_25_res(2);
	TEMP_26_res(3) <= TEMP_25_res(9) when stq_tail_i(2) else TEMP_25_res(3);
	TEMP_26_res(4) <= TEMP_25_res(0) when stq_tail_i(2) else TEMP_25_res(4);
	TEMP_26_res(5) <= TEMP_25_res(1) when stq_tail_i(2) else TEMP_25_res(5);
	TEMP_26_res(6) <= TEMP_25_res(2) when stq_tail_i(2) else TEMP_25_res(6);
	TEMP_26_res(7) <= TEMP_25_res(3) when stq_tail_i(2) else TEMP_25_res(7);
	TEMP_26_res(8) <= TEMP_25_res(4) when stq_tail_i(2) else TEMP_25_res(8);
	TEMP_26_res(9) <= TEMP_25_res(5) when stq_tail_i(2) else TEMP_25_res(9);
	-- Layer End
	TEMP_27_res(0) <= TEMP_26_res(8) when stq_tail_i(1) else TEMP_26_res(0);
	TEMP_27_res(1) <= TEMP_26_res(9) when stq_tail_i(1) else TEMP_26_res(1);
	TEMP_27_res(2) <= TEMP_26_res(0) when stq_tail_i(1) else TEMP_26_res(2);
	TEMP_27_res(3) <= TEMP_26_res(1) when stq_tail_i(1) else TEMP_26_res(3);
	TEMP_27_res(4) <= TEMP_26_res(2) when stq_tail_i(1) else TEMP_26_res(4);
	TEMP_27_res(5) <= TEMP_26_res(3) when stq_tail_i(1) else TEMP_26_res(5);
	TEMP_27_res(6) <= TEMP_26_res(4) when stq_tail_i(1) else TEMP_26_res(6);
	TEMP_27_res(7) <= TEMP_26_res(5) when stq_tail_i(1) else TEMP_26_res(7);
	TEMP_27_res(8) <= TEMP_26_res(6) when stq_tail_i(1) else TEMP_26_res(8);
	TEMP_27_res(9) <= TEMP_26_res(7) when stq_tail_i(1) else TEMP_26_res(9);
	-- Layer End
	ga_ls_order_temp_5(0) <= TEMP_27_res(9) when stq_tail_i(0) else TEMP_27_res(0);
	ga_ls_order_temp_5(1) <= TEMP_27_res(0) when stq_tail_i(0) else TEMP_27_res(1);
	ga_ls_order_temp_5(2) <= TEMP_27_res(1) when stq_tail_i(0) else TEMP_27_res(2);
	ga_ls_order_temp_5(3) <= TEMP_27_res(2) when stq_tail_i(0) else TEMP_27_res(3);
	ga_ls_order_temp_5(4) <= TEMP_27_res(3) when stq_tail_i(0) else TEMP_27_res(4);
	ga_ls_order_temp_5(5) <= TEMP_27_res(4) when stq_tail_i(0) else TEMP_27_res(5);
	ga_ls_order_temp_5(6) <= TEMP_27_res(5) when stq_tail_i(0) else TEMP_27_res(6);
	ga_ls_order_temp_5(7) <= TEMP_27_res(6) when stq_tail_i(0) else TEMP_27_res(7);
	ga_ls_order_temp_5(8) <= TEMP_27_res(7) when stq_tail_i(0) else TEMP_27_res(8);
	ga_ls_order_temp_5(9) <= TEMP_27_res(8) when stq_tail_i(0) else TEMP_27_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_6, ga_ls_order_rom_6, stq_tail)
	TEMP_28_res(0) <= ga_ls_order_rom_6(2) when stq_tail_i(3) else ga_ls_order_rom_6(0);
	TEMP_28_res(1) <= ga_ls_order_rom_6(3) when stq_tail_i(3) else ga_ls_order_rom_6(1);
	TEMP_28_res(2) <= ga_ls_order_rom_6(4) when stq_tail_i(3) else ga_ls_order_rom_6(2);
	TEMP_28_res(3) <= ga_ls_order_rom_6(5) when stq_tail_i(3) else ga_ls_order_rom_6(3);
	TEMP_28_res(4) <= ga_ls_order_rom_6(6) when stq_tail_i(3) else ga_ls_order_rom_6(4);
	TEMP_28_res(5) <= ga_ls_order_rom_6(7) when stq_tail_i(3) else ga_ls_order_rom_6(5);
	TEMP_28_res(6) <= ga_ls_order_rom_6(8) when stq_tail_i(3) else ga_ls_order_rom_6(6);
	TEMP_28_res(7) <= ga_ls_order_rom_6(9) when stq_tail_i(3) else ga_ls_order_rom_6(7);
	TEMP_28_res(8) <= ga_ls_order_rom_6(0) when stq_tail_i(3) else ga_ls_order_rom_6(8);
	TEMP_28_res(9) <= ga_ls_order_rom_6(1) when stq_tail_i(3) else ga_ls_order_rom_6(9);
	-- Layer End
	TEMP_29_res(0) <= TEMP_28_res(6) when stq_tail_i(2) else TEMP_28_res(0);
	TEMP_29_res(1) <= TEMP_28_res(7) when stq_tail_i(2) else TEMP_28_res(1);
	TEMP_29_res(2) <= TEMP_28_res(8) when stq_tail_i(2) else TEMP_28_res(2);
	TEMP_29_res(3) <= TEMP_28_res(9) when stq_tail_i(2) else TEMP_28_res(3);
	TEMP_29_res(4) <= TEMP_28_res(0) when stq_tail_i(2) else TEMP_28_res(4);
	TEMP_29_res(5) <= TEMP_28_res(1) when stq_tail_i(2) else TEMP_28_res(5);
	TEMP_29_res(6) <= TEMP_28_res(2) when stq_tail_i(2) else TEMP_28_res(6);
	TEMP_29_res(7) <= TEMP_28_res(3) when stq_tail_i(2) else TEMP_28_res(7);
	TEMP_29_res(8) <= TEMP_28_res(4) when stq_tail_i(2) else TEMP_28_res(8);
	TEMP_29_res(9) <= TEMP_28_res(5) when stq_tail_i(2) else TEMP_28_res(9);
	-- Layer End
	TEMP_30_res(0) <= TEMP_29_res(8) when stq_tail_i(1) else TEMP_29_res(0);
	TEMP_30_res(1) <= TEMP_29_res(9) when stq_tail_i(1) else TEMP_29_res(1);
	TEMP_30_res(2) <= TEMP_29_res(0) when stq_tail_i(1) else TEMP_29_res(2);
	TEMP_30_res(3) <= TEMP_29_res(1) when stq_tail_i(1) else TEMP_29_res(3);
	TEMP_30_res(4) <= TEMP_29_res(2) when stq_tail_i(1) else TEMP_29_res(4);
	TEMP_30_res(5) <= TEMP_29_res(3) when stq_tail_i(1) else TEMP_29_res(5);
	TEMP_30_res(6) <= TEMP_29_res(4) when stq_tail_i(1) else TEMP_29_res(6);
	TEMP_30_res(7) <= TEMP_29_res(5) when stq_tail_i(1) else TEMP_29_res(7);
	TEMP_30_res(8) <= TEMP_29_res(6) when stq_tail_i(1) else TEMP_29_res(8);
	TEMP_30_res(9) <= TEMP_29_res(7) when stq_tail_i(1) else TEMP_29_res(9);
	-- Layer End
	ga_ls_order_temp_6(0) <= TEMP_30_res(9) when stq_tail_i(0) else TEMP_30_res(0);
	ga_ls_order_temp_6(1) <= TEMP_30_res(0) when stq_tail_i(0) else TEMP_30_res(1);
	ga_ls_order_temp_6(2) <= TEMP_30_res(1) when stq_tail_i(0) else TEMP_30_res(2);
	ga_ls_order_temp_6(3) <= TEMP_30_res(2) when stq_tail_i(0) else TEMP_30_res(3);
	ga_ls_order_temp_6(4) <= TEMP_30_res(3) when stq_tail_i(0) else TEMP_30_res(4);
	ga_ls_order_temp_6(5) <= TEMP_30_res(4) when stq_tail_i(0) else TEMP_30_res(5);
	ga_ls_order_temp_6(6) <= TEMP_30_res(5) when stq_tail_i(0) else TEMP_30_res(6);
	ga_ls_order_temp_6(7) <= TEMP_30_res(6) when stq_tail_i(0) else TEMP_30_res(7);
	ga_ls_order_temp_6(8) <= TEMP_30_res(7) when stq_tail_i(0) else TEMP_30_res(8);
	ga_ls_order_temp_6(9) <= TEMP_30_res(8) when stq_tail_i(0) else TEMP_30_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_7, ga_ls_order_rom_7, stq_tail)
	TEMP_31_res(0) <= ga_ls_order_rom_7(2) when stq_tail_i(3) else ga_ls_order_rom_7(0);
	TEMP_31_res(1) <= ga_ls_order_rom_7(3) when stq_tail_i(3) else ga_ls_order_rom_7(1);
	TEMP_31_res(2) <= ga_ls_order_rom_7(4) when stq_tail_i(3) else ga_ls_order_rom_7(2);
	TEMP_31_res(3) <= ga_ls_order_rom_7(5) when stq_tail_i(3) else ga_ls_order_rom_7(3);
	TEMP_31_res(4) <= ga_ls_order_rom_7(6) when stq_tail_i(3) else ga_ls_order_rom_7(4);
	TEMP_31_res(5) <= ga_ls_order_rom_7(7) when stq_tail_i(3) else ga_ls_order_rom_7(5);
	TEMP_31_res(6) <= ga_ls_order_rom_7(8) when stq_tail_i(3) else ga_ls_order_rom_7(6);
	TEMP_31_res(7) <= ga_ls_order_rom_7(9) when stq_tail_i(3) else ga_ls_order_rom_7(7);
	TEMP_31_res(8) <= ga_ls_order_rom_7(0) when stq_tail_i(3) else ga_ls_order_rom_7(8);
	TEMP_31_res(9) <= ga_ls_order_rom_7(1) when stq_tail_i(3) else ga_ls_order_rom_7(9);
	-- Layer End
	TEMP_32_res(0) <= TEMP_31_res(6) when stq_tail_i(2) else TEMP_31_res(0);
	TEMP_32_res(1) <= TEMP_31_res(7) when stq_tail_i(2) else TEMP_31_res(1);
	TEMP_32_res(2) <= TEMP_31_res(8) when stq_tail_i(2) else TEMP_31_res(2);
	TEMP_32_res(3) <= TEMP_31_res(9) when stq_tail_i(2) else TEMP_31_res(3);
	TEMP_32_res(4) <= TEMP_31_res(0) when stq_tail_i(2) else TEMP_31_res(4);
	TEMP_32_res(5) <= TEMP_31_res(1) when stq_tail_i(2) else TEMP_31_res(5);
	TEMP_32_res(6) <= TEMP_31_res(2) when stq_tail_i(2) else TEMP_31_res(6);
	TEMP_32_res(7) <= TEMP_31_res(3) when stq_tail_i(2) else TEMP_31_res(7);
	TEMP_32_res(8) <= TEMP_31_res(4) when stq_tail_i(2) else TEMP_31_res(8);
	TEMP_32_res(9) <= TEMP_31_res(5) when stq_tail_i(2) else TEMP_31_res(9);
	-- Layer End
	TEMP_33_res(0) <= TEMP_32_res(8) when stq_tail_i(1) else TEMP_32_res(0);
	TEMP_33_res(1) <= TEMP_32_res(9) when stq_tail_i(1) else TEMP_32_res(1);
	TEMP_33_res(2) <= TEMP_32_res(0) when stq_tail_i(1) else TEMP_32_res(2);
	TEMP_33_res(3) <= TEMP_32_res(1) when stq_tail_i(1) else TEMP_32_res(3);
	TEMP_33_res(4) <= TEMP_32_res(2) when stq_tail_i(1) else TEMP_32_res(4);
	TEMP_33_res(5) <= TEMP_32_res(3) when stq_tail_i(1) else TEMP_32_res(5);
	TEMP_33_res(6) <= TEMP_32_res(4) when stq_tail_i(1) else TEMP_32_res(6);
	TEMP_33_res(7) <= TEMP_32_res(5) when stq_tail_i(1) else TEMP_32_res(7);
	TEMP_33_res(8) <= TEMP_32_res(6) when stq_tail_i(1) else TEMP_32_res(8);
	TEMP_33_res(9) <= TEMP_32_res(7) when stq_tail_i(1) else TEMP_32_res(9);
	-- Layer End
	ga_ls_order_temp_7(0) <= TEMP_33_res(9) when stq_tail_i(0) else TEMP_33_res(0);
	ga_ls_order_temp_7(1) <= TEMP_33_res(0) when stq_tail_i(0) else TEMP_33_res(1);
	ga_ls_order_temp_7(2) <= TEMP_33_res(1) when stq_tail_i(0) else TEMP_33_res(2);
	ga_ls_order_temp_7(3) <= TEMP_33_res(2) when stq_tail_i(0) else TEMP_33_res(3);
	ga_ls_order_temp_7(4) <= TEMP_33_res(3) when stq_tail_i(0) else TEMP_33_res(4);
	ga_ls_order_temp_7(5) <= TEMP_33_res(4) when stq_tail_i(0) else TEMP_33_res(5);
	ga_ls_order_temp_7(6) <= TEMP_33_res(5) when stq_tail_i(0) else TEMP_33_res(6);
	ga_ls_order_temp_7(7) <= TEMP_33_res(6) when stq_tail_i(0) else TEMP_33_res(7);
	ga_ls_order_temp_7(8) <= TEMP_33_res(7) when stq_tail_i(0) else TEMP_33_res(8);
	ga_ls_order_temp_7(9) <= TEMP_33_res(8) when stq_tail_i(0) else TEMP_33_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_8, ga_ls_order_rom_8, stq_tail)
	TEMP_34_res(0) <= ga_ls_order_rom_8(2) when stq_tail_i(3) else ga_ls_order_rom_8(0);
	TEMP_34_res(1) <= ga_ls_order_rom_8(3) when stq_tail_i(3) else ga_ls_order_rom_8(1);
	TEMP_34_res(2) <= ga_ls_order_rom_8(4) when stq_tail_i(3) else ga_ls_order_rom_8(2);
	TEMP_34_res(3) <= ga_ls_order_rom_8(5) when stq_tail_i(3) else ga_ls_order_rom_8(3);
	TEMP_34_res(4) <= ga_ls_order_rom_8(6) when stq_tail_i(3) else ga_ls_order_rom_8(4);
	TEMP_34_res(5) <= ga_ls_order_rom_8(7) when stq_tail_i(3) else ga_ls_order_rom_8(5);
	TEMP_34_res(6) <= ga_ls_order_rom_8(8) when stq_tail_i(3) else ga_ls_order_rom_8(6);
	TEMP_34_res(7) <= ga_ls_order_rom_8(9) when stq_tail_i(3) else ga_ls_order_rom_8(7);
	TEMP_34_res(8) <= ga_ls_order_rom_8(0) when stq_tail_i(3) else ga_ls_order_rom_8(8);
	TEMP_34_res(9) <= ga_ls_order_rom_8(1) when stq_tail_i(3) else ga_ls_order_rom_8(9);
	-- Layer End
	TEMP_35_res(0) <= TEMP_34_res(6) when stq_tail_i(2) else TEMP_34_res(0);
	TEMP_35_res(1) <= TEMP_34_res(7) when stq_tail_i(2) else TEMP_34_res(1);
	TEMP_35_res(2) <= TEMP_34_res(8) when stq_tail_i(2) else TEMP_34_res(2);
	TEMP_35_res(3) <= TEMP_34_res(9) when stq_tail_i(2) else TEMP_34_res(3);
	TEMP_35_res(4) <= TEMP_34_res(0) when stq_tail_i(2) else TEMP_34_res(4);
	TEMP_35_res(5) <= TEMP_34_res(1) when stq_tail_i(2) else TEMP_34_res(5);
	TEMP_35_res(6) <= TEMP_34_res(2) when stq_tail_i(2) else TEMP_34_res(6);
	TEMP_35_res(7) <= TEMP_34_res(3) when stq_tail_i(2) else TEMP_34_res(7);
	TEMP_35_res(8) <= TEMP_34_res(4) when stq_tail_i(2) else TEMP_34_res(8);
	TEMP_35_res(9) <= TEMP_34_res(5) when stq_tail_i(2) else TEMP_34_res(9);
	-- Layer End
	TEMP_36_res(0) <= TEMP_35_res(8) when stq_tail_i(1) else TEMP_35_res(0);
	TEMP_36_res(1) <= TEMP_35_res(9) when stq_tail_i(1) else TEMP_35_res(1);
	TEMP_36_res(2) <= TEMP_35_res(0) when stq_tail_i(1) else TEMP_35_res(2);
	TEMP_36_res(3) <= TEMP_35_res(1) when stq_tail_i(1) else TEMP_35_res(3);
	TEMP_36_res(4) <= TEMP_35_res(2) when stq_tail_i(1) else TEMP_35_res(4);
	TEMP_36_res(5) <= TEMP_35_res(3) when stq_tail_i(1) else TEMP_35_res(5);
	TEMP_36_res(6) <= TEMP_35_res(4) when stq_tail_i(1) else TEMP_35_res(6);
	TEMP_36_res(7) <= TEMP_35_res(5) when stq_tail_i(1) else TEMP_35_res(7);
	TEMP_36_res(8) <= TEMP_35_res(6) when stq_tail_i(1) else TEMP_35_res(8);
	TEMP_36_res(9) <= TEMP_35_res(7) when stq_tail_i(1) else TEMP_35_res(9);
	-- Layer End
	ga_ls_order_temp_8(0) <= TEMP_36_res(9) when stq_tail_i(0) else TEMP_36_res(0);
	ga_ls_order_temp_8(1) <= TEMP_36_res(0) when stq_tail_i(0) else TEMP_36_res(1);
	ga_ls_order_temp_8(2) <= TEMP_36_res(1) when stq_tail_i(0) else TEMP_36_res(2);
	ga_ls_order_temp_8(3) <= TEMP_36_res(2) when stq_tail_i(0) else TEMP_36_res(3);
	ga_ls_order_temp_8(4) <= TEMP_36_res(3) when stq_tail_i(0) else TEMP_36_res(4);
	ga_ls_order_temp_8(5) <= TEMP_36_res(4) when stq_tail_i(0) else TEMP_36_res(5);
	ga_ls_order_temp_8(6) <= TEMP_36_res(5) when stq_tail_i(0) else TEMP_36_res(6);
	ga_ls_order_temp_8(7) <= TEMP_36_res(6) when stq_tail_i(0) else TEMP_36_res(7);
	ga_ls_order_temp_8(8) <= TEMP_36_res(7) when stq_tail_i(0) else TEMP_36_res(8);
	ga_ls_order_temp_8(9) <= TEMP_36_res(8) when stq_tail_i(0) else TEMP_36_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order_temp_9, ga_ls_order_rom_9, stq_tail)
	TEMP_37_res(0) <= ga_ls_order_rom_9(2) when stq_tail_i(3) else ga_ls_order_rom_9(0);
	TEMP_37_res(1) <= ga_ls_order_rom_9(3) when stq_tail_i(3) else ga_ls_order_rom_9(1);
	TEMP_37_res(2) <= ga_ls_order_rom_9(4) when stq_tail_i(3) else ga_ls_order_rom_9(2);
	TEMP_37_res(3) <= ga_ls_order_rom_9(5) when stq_tail_i(3) else ga_ls_order_rom_9(3);
	TEMP_37_res(4) <= ga_ls_order_rom_9(6) when stq_tail_i(3) else ga_ls_order_rom_9(4);
	TEMP_37_res(5) <= ga_ls_order_rom_9(7) when stq_tail_i(3) else ga_ls_order_rom_9(5);
	TEMP_37_res(6) <= ga_ls_order_rom_9(8) when stq_tail_i(3) else ga_ls_order_rom_9(6);
	TEMP_37_res(7) <= ga_ls_order_rom_9(9) when stq_tail_i(3) else ga_ls_order_rom_9(7);
	TEMP_37_res(8) <= ga_ls_order_rom_9(0) when stq_tail_i(3) else ga_ls_order_rom_9(8);
	TEMP_37_res(9) <= ga_ls_order_rom_9(1) when stq_tail_i(3) else ga_ls_order_rom_9(9);
	-- Layer End
	TEMP_38_res(0) <= TEMP_37_res(6) when stq_tail_i(2) else TEMP_37_res(0);
	TEMP_38_res(1) <= TEMP_37_res(7) when stq_tail_i(2) else TEMP_37_res(1);
	TEMP_38_res(2) <= TEMP_37_res(8) when stq_tail_i(2) else TEMP_37_res(2);
	TEMP_38_res(3) <= TEMP_37_res(9) when stq_tail_i(2) else TEMP_37_res(3);
	TEMP_38_res(4) <= TEMP_37_res(0) when stq_tail_i(2) else TEMP_37_res(4);
	TEMP_38_res(5) <= TEMP_37_res(1) when stq_tail_i(2) else TEMP_37_res(5);
	TEMP_38_res(6) <= TEMP_37_res(2) when stq_tail_i(2) else TEMP_37_res(6);
	TEMP_38_res(7) <= TEMP_37_res(3) when stq_tail_i(2) else TEMP_37_res(7);
	TEMP_38_res(8) <= TEMP_37_res(4) when stq_tail_i(2) else TEMP_37_res(8);
	TEMP_38_res(9) <= TEMP_37_res(5) when stq_tail_i(2) else TEMP_37_res(9);
	-- Layer End
	TEMP_39_res(0) <= TEMP_38_res(8) when stq_tail_i(1) else TEMP_38_res(0);
	TEMP_39_res(1) <= TEMP_38_res(9) when stq_tail_i(1) else TEMP_38_res(1);
	TEMP_39_res(2) <= TEMP_38_res(0) when stq_tail_i(1) else TEMP_38_res(2);
	TEMP_39_res(3) <= TEMP_38_res(1) when stq_tail_i(1) else TEMP_38_res(3);
	TEMP_39_res(4) <= TEMP_38_res(2) when stq_tail_i(1) else TEMP_38_res(4);
	TEMP_39_res(5) <= TEMP_38_res(3) when stq_tail_i(1) else TEMP_38_res(5);
	TEMP_39_res(6) <= TEMP_38_res(4) when stq_tail_i(1) else TEMP_38_res(6);
	TEMP_39_res(7) <= TEMP_38_res(5) when stq_tail_i(1) else TEMP_38_res(7);
	TEMP_39_res(8) <= TEMP_38_res(6) when stq_tail_i(1) else TEMP_38_res(8);
	TEMP_39_res(9) <= TEMP_38_res(7) when stq_tail_i(1) else TEMP_38_res(9);
	-- Layer End
	ga_ls_order_temp_9(0) <= TEMP_39_res(9) when stq_tail_i(0) else TEMP_39_res(0);
	ga_ls_order_temp_9(1) <= TEMP_39_res(0) when stq_tail_i(0) else TEMP_39_res(1);
	ga_ls_order_temp_9(2) <= TEMP_39_res(1) when stq_tail_i(0) else TEMP_39_res(2);
	ga_ls_order_temp_9(3) <= TEMP_39_res(2) when stq_tail_i(0) else TEMP_39_res(3);
	ga_ls_order_temp_9(4) <= TEMP_39_res(3) when stq_tail_i(0) else TEMP_39_res(4);
	ga_ls_order_temp_9(5) <= TEMP_39_res(4) when stq_tail_i(0) else TEMP_39_res(5);
	ga_ls_order_temp_9(6) <= TEMP_39_res(5) when stq_tail_i(0) else TEMP_39_res(6);
	ga_ls_order_temp_9(7) <= TEMP_39_res(6) when stq_tail_i(0) else TEMP_39_res(7);
	ga_ls_order_temp_9(8) <= TEMP_39_res(7) when stq_tail_i(0) else TEMP_39_res(8);
	ga_ls_order_temp_9(9) <= TEMP_39_res(8) when stq_tail_i(0) else TEMP_39_res(9);
	-- Shifter End

	-- Shifter Begin
	-- CyclicLeftShift(ga_ls_order, ga_ls_order_temp, ldq_tail)
	TEMP_40_res_0 <= ga_ls_order_temp_2 when ldq_tail_i(3) else ga_ls_order_temp_0;
	TEMP_40_res_1 <= ga_ls_order_temp_3 when ldq_tail_i(3) else ga_ls_order_temp_1;
	TEMP_40_res_2 <= ga_ls_order_temp_4 when ldq_tail_i(3) else ga_ls_order_temp_2;
	TEMP_40_res_3 <= ga_ls_order_temp_5 when ldq_tail_i(3) else ga_ls_order_temp_3;
	TEMP_40_res_4 <= ga_ls_order_temp_6 when ldq_tail_i(3) else ga_ls_order_temp_4;
	TEMP_40_res_5 <= ga_ls_order_temp_7 when ldq_tail_i(3) else ga_ls_order_temp_5;
	TEMP_40_res_6 <= ga_ls_order_temp_8 when ldq_tail_i(3) else ga_ls_order_temp_6;
	TEMP_40_res_7 <= ga_ls_order_temp_9 when ldq_tail_i(3) else ga_ls_order_temp_7;
	TEMP_40_res_8 <= ga_ls_order_temp_0 when ldq_tail_i(3) else ga_ls_order_temp_8;
	TEMP_40_res_9 <= ga_ls_order_temp_1 when ldq_tail_i(3) else ga_ls_order_temp_9;
	-- Layer End
	TEMP_41_res_0 <= TEMP_40_res_6 when ldq_tail_i(2) else TEMP_40_res_0;
	TEMP_41_res_1 <= TEMP_40_res_7 when ldq_tail_i(2) else TEMP_40_res_1;
	TEMP_41_res_2 <= TEMP_40_res_8 when ldq_tail_i(2) else TEMP_40_res_2;
	TEMP_41_res_3 <= TEMP_40_res_9 when ldq_tail_i(2) else TEMP_40_res_3;
	TEMP_41_res_4 <= TEMP_40_res_0 when ldq_tail_i(2) else TEMP_40_res_4;
	TEMP_41_res_5 <= TEMP_40_res_1 when ldq_tail_i(2) else TEMP_40_res_5;
	TEMP_41_res_6 <= TEMP_40_res_2 when ldq_tail_i(2) else TEMP_40_res_6;
	TEMP_41_res_7 <= TEMP_40_res_3 when ldq_tail_i(2) else TEMP_40_res_7;
	TEMP_41_res_8 <= TEMP_40_res_4 when ldq_tail_i(2) else TEMP_40_res_8;
	TEMP_41_res_9 <= TEMP_40_res_5 when ldq_tail_i(2) else TEMP_40_res_9;
	-- Layer End
	TEMP_42_res_0 <= TEMP_41_res_8 when ldq_tail_i(1) else TEMP_41_res_0;
	TEMP_42_res_1 <= TEMP_41_res_9 when ldq_tail_i(1) else TEMP_41_res_1;
	TEMP_42_res_2 <= TEMP_41_res_0 when ldq_tail_i(1) else TEMP_41_res_2;
	TEMP_42_res_3 <= TEMP_41_res_1 when ldq_tail_i(1) else TEMP_41_res_3;
	TEMP_42_res_4 <= TEMP_41_res_2 when ldq_tail_i(1) else TEMP_41_res_4;
	TEMP_42_res_5 <= TEMP_41_res_3 when ldq_tail_i(1) else TEMP_41_res_5;
	TEMP_42_res_6 <= TEMP_41_res_4 when ldq_tail_i(1) else TEMP_41_res_6;
	TEMP_42_res_7 <= TEMP_41_res_5 when ldq_tail_i(1) else TEMP_41_res_7;
	TEMP_42_res_8 <= TEMP_41_res_6 when ldq_tail_i(1) else TEMP_41_res_8;
	TEMP_42_res_9 <= TEMP_41_res_7 when ldq_tail_i(1) else TEMP_41_res_9;
	-- Layer End
	ga_ls_order_0_o <= TEMP_42_res_9 when ldq_tail_i(0) else TEMP_42_res_0;
	ga_ls_order_1_o <= TEMP_42_res_0 when ldq_tail_i(0) else TEMP_42_res_1;
	ga_ls_order_2_o <= TEMP_42_res_1 when ldq_tail_i(0) else TEMP_42_res_2;
	ga_ls_order_3_o <= TEMP_42_res_2 when ldq_tail_i(0) else TEMP_42_res_3;
	ga_ls_order_4_o <= TEMP_42_res_3 when ldq_tail_i(0) else TEMP_42_res_4;
	ga_ls_order_5_o <= TEMP_42_res_4 when ldq_tail_i(0) else TEMP_42_res_5;
	ga_ls_order_6_o <= TEMP_42_res_5 when ldq_tail_i(0) else TEMP_42_res_6;
	ga_ls_order_7_o <= TEMP_42_res_6 when ldq_tail_i(0) else TEMP_42_res_7;
	ga_ls_order_8_o <= TEMP_42_res_7 when ldq_tail_i(0) else TEMP_42_res_8;
	ga_ls_order_9_o <= TEMP_42_res_8 when ldq_tail_i(0) else TEMP_42_res_9;
	-- Shifter End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_lda is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_payload_2_o : out std_logic_vector(9 downto 0);
		entry_payload_3_o : out std_logic_vector(9 downto 0);
		entry_payload_4_o : out std_logic_vector(9 downto 0);
		entry_payload_5_o : out std_logic_vector(9 downto 0);
		entry_payload_6_o : out std_logic_vector(9 downto 0);
		entry_payload_7_o : out std_logic_vector(9 downto 0);
		entry_payload_8_o : out std_logic_vector(9 downto 0);
		entry_payload_9_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(9 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_lda is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(9 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_11_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal TEMP_14_double_in_0 : std_logic_vector(19 downto 0);
	signal TEMP_14_double_out_0 : std_logic_vector(19 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "0000000000";
	entry_payload_2_o <= TEMP_3_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "0000000000";
	entry_payload_3_o <= TEMP_4_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "0000000000";
	entry_payload_4_o <= TEMP_5_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "0000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "0000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_11_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_11_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_11_res_2 <= entry_waiting_for_port_2;
	TEMP_11_res_3 <= entry_waiting_for_port_3;
	TEMP_11_res_4 <= entry_waiting_for_port_4;
	TEMP_11_res_5 <= entry_waiting_for_port_5;
	TEMP_11_res_6 <= entry_waiting_for_port_6;
	TEMP_11_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_4;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_5;
	TEMP_12_res_2 <= TEMP_11_res_2 or TEMP_11_res_6;
	TEMP_12_res_3 <= TEMP_11_res_3 or TEMP_11_res_7;
	-- Layer End
	TEMP_13_res_0 <= TEMP_12_res_0 or TEMP_12_res_2;
	TEMP_13_res_1 <= TEMP_12_res_1 or TEMP_12_res_3;
	-- Layer End
	port_ready_vec <= TEMP_13_res_0 or TEMP_13_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_14_double_in_0(0) <= entry_port_options_0(0);
	TEMP_14_double_in_0(10) <= entry_port_options_0(0);
	TEMP_14_double_in_0(1) <= entry_port_options_1(0);
	TEMP_14_double_in_0(11) <= entry_port_options_1(0);
	TEMP_14_double_in_0(2) <= entry_port_options_2(0);
	TEMP_14_double_in_0(12) <= entry_port_options_2(0);
	TEMP_14_double_in_0(3) <= entry_port_options_3(0);
	TEMP_14_double_in_0(13) <= entry_port_options_3(0);
	TEMP_14_double_in_0(4) <= entry_port_options_4(0);
	TEMP_14_double_in_0(14) <= entry_port_options_4(0);
	TEMP_14_double_in_0(5) <= entry_port_options_5(0);
	TEMP_14_double_in_0(15) <= entry_port_options_5(0);
	TEMP_14_double_in_0(6) <= entry_port_options_6(0);
	TEMP_14_double_in_0(16) <= entry_port_options_6(0);
	TEMP_14_double_in_0(7) <= entry_port_options_7(0);
	TEMP_14_double_in_0(17) <= entry_port_options_7(0);
	TEMP_14_double_in_0(8) <= entry_port_options_8(0);
	TEMP_14_double_in_0(18) <= entry_port_options_8(0);
	TEMP_14_double_in_0(9) <= entry_port_options_9(0);
	TEMP_14_double_in_0(19) <= entry_port_options_9(0);
	TEMP_14_double_out_0 <= TEMP_14_double_in_0 and not std_logic_vector( unsigned( TEMP_14_double_in_0 ) - unsigned( "0000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_14_double_out_0(0) or TEMP_14_double_out_0(10);
	entry_port_transfer_1(0) <= TEMP_14_double_out_0(1) or TEMP_14_double_out_0(11);
	entry_port_transfer_2(0) <= TEMP_14_double_out_0(2) or TEMP_14_double_out_0(12);
	entry_port_transfer_3(0) <= TEMP_14_double_out_0(3) or TEMP_14_double_out_0(13);
	entry_port_transfer_4(0) <= TEMP_14_double_out_0(4) or TEMP_14_double_out_0(14);
	entry_port_transfer_5(0) <= TEMP_14_double_out_0(5) or TEMP_14_double_out_0(15);
	entry_port_transfer_6(0) <= TEMP_14_double_out_0(6) or TEMP_14_double_out_0(16);
	entry_port_transfer_7(0) <= TEMP_14_double_out_0(7) or TEMP_14_double_out_0(17);
	entry_port_transfer_8(0) <= TEMP_14_double_out_0(8) or TEMP_14_double_out_0(18);
	entry_port_transfer_9(0) <= TEMP_14_double_out_0(9) or TEMP_14_double_out_0(19);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_ldd is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_o : out std_logic_vector(31 downto 0);
		port_valid_0_o : out std_logic;
		port_ready_0_i : in std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_0_i : in std_logic_vector(31 downto 0);
		entry_payload_1_i : in std_logic_vector(31 downto 0);
		entry_payload_2_i : in std_logic_vector(31 downto 0);
		entry_payload_3_i : in std_logic_vector(31 downto 0);
		entry_payload_4_i : in std_logic_vector(31 downto 0);
		entry_payload_5_i : in std_logic_vector(31 downto 0);
		entry_payload_6_i : in std_logic_vector(31 downto 0);
		entry_payload_7_i : in std_logic_vector(31 downto 0);
		entry_payload_8_i : in std_logic_vector(31 downto 0);
		entry_payload_9_i : in std_logic_vector(31 downto 0);
		entry_reset_0_o : out std_logic;
		entry_reset_1_o : out std_logic;
		entry_reset_2_o : out std_logic;
		entry_reset_3_o : out std_logic;
		entry_reset_4_o : out std_logic;
		entry_reset_5_o : out std_logic;
		entry_reset_6_o : out std_logic;
		entry_reset_7_o : out std_logic;
		entry_reset_8_o : out std_logic;
		entry_reset_9_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(9 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_ldd is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_allocated_for_port_9 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_0 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_1 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_2 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_3 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_4 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_5 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_6 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_7 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_8 : std_logic_vector(0 downto 0);
	signal oldest_entry_allocated_per_port_9 : std_logic_vector(0 downto 0);
	signal TEMP_1_double_in_0 : std_logic_vector(19 downto 0);
	signal TEMP_1_double_out_0 : std_logic_vector(19 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_3_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_4_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_res_1 : std_logic_vector(31 downto 0);
	signal entry_waiting_for_port_valid_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_valid_9 : std_logic_vector(0 downto 0);
	signal port_valid_vec : std_logic_vector(0 downto 0);
	signal TEMP_6_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_6_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_7_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_8_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	entry_allocated_for_port_0 <= entry_port_idx_oh_0 when entry_alloc_0_i else "0";
	entry_allocated_for_port_1 <= entry_port_idx_oh_1 when entry_alloc_1_i else "0";
	entry_allocated_for_port_2 <= entry_port_idx_oh_2 when entry_alloc_2_i else "0";
	entry_allocated_for_port_3 <= entry_port_idx_oh_3 when entry_alloc_3_i else "0";
	entry_allocated_for_port_4 <= entry_port_idx_oh_4 when entry_alloc_4_i else "0";
	entry_allocated_for_port_5 <= entry_port_idx_oh_5 when entry_alloc_5_i else "0";
	entry_allocated_for_port_6 <= entry_port_idx_oh_6 when entry_alloc_6_i else "0";
	entry_allocated_for_port_7 <= entry_port_idx_oh_7 when entry_alloc_7_i else "0";
	entry_allocated_for_port_8 <= entry_port_idx_oh_8 when entry_alloc_8_i else "0";
	entry_allocated_for_port_9 <= entry_port_idx_oh_9 when entry_alloc_9_i else "0";
	-- Priority Masking Begin
	-- CyclicPriorityMask(oldest_entry_allocated_per_port, entry_allocated_for_port, queue_head_oh)
	TEMP_1_double_in_0(0) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(10) <= entry_allocated_for_port_0(0);
	TEMP_1_double_in_0(1) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(11) <= entry_allocated_for_port_1(0);
	TEMP_1_double_in_0(2) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(12) <= entry_allocated_for_port_2(0);
	TEMP_1_double_in_0(3) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(13) <= entry_allocated_for_port_3(0);
	TEMP_1_double_in_0(4) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(14) <= entry_allocated_for_port_4(0);
	TEMP_1_double_in_0(5) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(15) <= entry_allocated_for_port_5(0);
	TEMP_1_double_in_0(6) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(16) <= entry_allocated_for_port_6(0);
	TEMP_1_double_in_0(7) <= entry_allocated_for_port_7(0);
	TEMP_1_double_in_0(17) <= entry_allocated_for_port_7(0);
	TEMP_1_double_in_0(8) <= entry_allocated_for_port_8(0);
	TEMP_1_double_in_0(18) <= entry_allocated_for_port_8(0);
	TEMP_1_double_in_0(9) <= entry_allocated_for_port_9(0);
	TEMP_1_double_in_0(19) <= entry_allocated_for_port_9(0);
	TEMP_1_double_out_0 <= TEMP_1_double_in_0 and not std_logic_vector( unsigned( TEMP_1_double_in_0 ) - unsigned( "0000000000" & queue_head_oh_i ) );
	oldest_entry_allocated_per_port_0(0) <= TEMP_1_double_out_0(0) or TEMP_1_double_out_0(10);
	oldest_entry_allocated_per_port_1(0) <= TEMP_1_double_out_0(1) or TEMP_1_double_out_0(11);
	oldest_entry_allocated_per_port_2(0) <= TEMP_1_double_out_0(2) or TEMP_1_double_out_0(12);
	oldest_entry_allocated_per_port_3(0) <= TEMP_1_double_out_0(3) or TEMP_1_double_out_0(13);
	oldest_entry_allocated_per_port_4(0) <= TEMP_1_double_out_0(4) or TEMP_1_double_out_0(14);
	oldest_entry_allocated_per_port_5(0) <= TEMP_1_double_out_0(5) or TEMP_1_double_out_0(15);
	oldest_entry_allocated_per_port_6(0) <= TEMP_1_double_out_0(6) or TEMP_1_double_out_0(16);
	oldest_entry_allocated_per_port_7(0) <= TEMP_1_double_out_0(7) or TEMP_1_double_out_0(17);
	oldest_entry_allocated_per_port_8(0) <= TEMP_1_double_out_0(8) or TEMP_1_double_out_0(18);
	oldest_entry_allocated_per_port_9(0) <= TEMP_1_double_out_0(9) or TEMP_1_double_out_0(19);
	-- Priority Masking End

	-- Mux1H Begin
	-- Mux1H(port_payload_0, entry_payload, oldest_entry_allocated_per_port)
	TEMP_2_mux_0 <= entry_payload_0_i when oldest_entry_allocated_per_port_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_1 <= entry_payload_1_i when oldest_entry_allocated_per_port_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_2 <= entry_payload_2_i when oldest_entry_allocated_per_port_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_3 <= entry_payload_3_i when oldest_entry_allocated_per_port_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_4 <= entry_payload_4_i when oldest_entry_allocated_per_port_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_5 <= entry_payload_5_i when oldest_entry_allocated_per_port_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_6 <= entry_payload_6_i when oldest_entry_allocated_per_port_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_7 <= entry_payload_7_i when oldest_entry_allocated_per_port_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_8 <= entry_payload_8_i when oldest_entry_allocated_per_port_8(0) = '1' else "00000000000000000000000000000000";
	TEMP_2_mux_9 <= entry_payload_9_i when oldest_entry_allocated_per_port_9(0) = '1' else "00000000000000000000000000000000";
	TEMP_3_res_0 <= TEMP_2_mux_0 or TEMP_2_mux_8;
	TEMP_3_res_1 <= TEMP_2_mux_1 or TEMP_2_mux_9;
	TEMP_3_res_2 <= TEMP_2_mux_2;
	TEMP_3_res_3 <= TEMP_2_mux_3;
	TEMP_3_res_4 <= TEMP_2_mux_4;
	TEMP_3_res_5 <= TEMP_2_mux_5;
	TEMP_3_res_6 <= TEMP_2_mux_6;
	TEMP_3_res_7 <= TEMP_2_mux_7;
	-- Layer End
	TEMP_4_res_0 <= TEMP_3_res_0 or TEMP_3_res_4;
	TEMP_4_res_1 <= TEMP_3_res_1 or TEMP_3_res_5;
	TEMP_4_res_2 <= TEMP_3_res_2 or TEMP_3_res_6;
	TEMP_4_res_3 <= TEMP_3_res_3 or TEMP_3_res_7;
	-- Layer End
	TEMP_5_res_0 <= TEMP_4_res_0 or TEMP_4_res_2;
	TEMP_5_res_1 <= TEMP_4_res_1 or TEMP_4_res_3;
	-- Layer End
	port_payload_0_o <= TEMP_5_res_0 or TEMP_5_res_1;
	-- Mux1H End

	entry_waiting_for_port_valid_0 <= oldest_entry_allocated_per_port_0 when entry_payload_valid_0_i else "0";
	entry_waiting_for_port_valid_1 <= oldest_entry_allocated_per_port_1 when entry_payload_valid_1_i else "0";
	entry_waiting_for_port_valid_2 <= oldest_entry_allocated_per_port_2 when entry_payload_valid_2_i else "0";
	entry_waiting_for_port_valid_3 <= oldest_entry_allocated_per_port_3 when entry_payload_valid_3_i else "0";
	entry_waiting_for_port_valid_4 <= oldest_entry_allocated_per_port_4 when entry_payload_valid_4_i else "0";
	entry_waiting_for_port_valid_5 <= oldest_entry_allocated_per_port_5 when entry_payload_valid_5_i else "0";
	entry_waiting_for_port_valid_6 <= oldest_entry_allocated_per_port_6 when entry_payload_valid_6_i else "0";
	entry_waiting_for_port_valid_7 <= oldest_entry_allocated_per_port_7 when entry_payload_valid_7_i else "0";
	entry_waiting_for_port_valid_8 <= oldest_entry_allocated_per_port_8 when entry_payload_valid_8_i else "0";
	entry_waiting_for_port_valid_9 <= oldest_entry_allocated_per_port_9 when entry_payload_valid_9_i else "0";
	-- Reduction Begin
	-- Reduce(port_valid_vec, entry_waiting_for_port_valid, or)
	TEMP_6_res_0 <= entry_waiting_for_port_valid_0 or entry_waiting_for_port_valid_8;
	TEMP_6_res_1 <= entry_waiting_for_port_valid_1 or entry_waiting_for_port_valid_9;
	TEMP_6_res_2 <= entry_waiting_for_port_valid_2;
	TEMP_6_res_3 <= entry_waiting_for_port_valid_3;
	TEMP_6_res_4 <= entry_waiting_for_port_valid_4;
	TEMP_6_res_5 <= entry_waiting_for_port_valid_5;
	TEMP_6_res_6 <= entry_waiting_for_port_valid_6;
	TEMP_6_res_7 <= entry_waiting_for_port_valid_7;
	-- Layer End
	TEMP_7_res_0 <= TEMP_6_res_0 or TEMP_6_res_4;
	TEMP_7_res_1 <= TEMP_6_res_1 or TEMP_6_res_5;
	TEMP_7_res_2 <= TEMP_6_res_2 or TEMP_6_res_6;
	TEMP_7_res_3 <= TEMP_6_res_3 or TEMP_6_res_7;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_0 or TEMP_7_res_2;
	TEMP_8_res_1 <= TEMP_7_res_1 or TEMP_7_res_3;
	-- Layer End
	port_valid_vec <= TEMP_8_res_0 or TEMP_8_res_1;
	-- Reduction End

	port_valid_0_o <= port_valid_vec(0);
	entry_port_transfer_0(0) <= entry_waiting_for_port_valid_0(0) and port_ready_0_i;
	entry_port_transfer_1(0) <= entry_waiting_for_port_valid_1(0) and port_ready_0_i;
	entry_port_transfer_2(0) <= entry_waiting_for_port_valid_2(0) and port_ready_0_i;
	entry_port_transfer_3(0) <= entry_waiting_for_port_valid_3(0) and port_ready_0_i;
	entry_port_transfer_4(0) <= entry_waiting_for_port_valid_4(0) and port_ready_0_i;
	entry_port_transfer_5(0) <= entry_waiting_for_port_valid_5(0) and port_ready_0_i;
	entry_port_transfer_6(0) <= entry_waiting_for_port_valid_6(0) and port_ready_0_i;
	entry_port_transfer_7(0) <= entry_waiting_for_port_valid_7(0) and port_ready_0_i;
	entry_port_transfer_8(0) <= entry_waiting_for_port_valid_8(0) and port_ready_0_i;
	entry_port_transfer_9(0) <= entry_waiting_for_port_valid_9(0) and port_ready_0_i;
	-- Reduction Begin
	-- Reduce(entry_reset_0, entry_port_transfer_0, or)
	entry_reset_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_1, entry_port_transfer_1, or)
	entry_reset_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_2, entry_port_transfer_2, or)
	entry_reset_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_3, entry_port_transfer_3, or)
	entry_reset_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_4, entry_port_transfer_4, or)
	entry_reset_4_o <= entry_port_transfer_4(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_5, entry_port_transfer_5, or)
	entry_reset_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_6, entry_port_transfer_6, or)
	entry_reset_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_7, entry_port_transfer_7, or)
	entry_reset_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_8, entry_port_transfer_8, or)
	entry_reset_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_reset_9, entry_port_transfer_9, or)
	entry_reset_9_o <= entry_port_transfer_9(0);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_sta is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(9 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(9 downto 0);
		entry_payload_1_o : out std_logic_vector(9 downto 0);
		entry_payload_2_o : out std_logic_vector(9 downto 0);
		entry_payload_3_o : out std_logic_vector(9 downto 0);
		entry_payload_4_o : out std_logic_vector(9 downto 0);
		entry_payload_5_o : out std_logic_vector(9 downto 0);
		entry_payload_6_o : out std_logic_vector(9 downto 0);
		entry_payload_7_o : out std_logic_vector(9 downto 0);
		entry_payload_8_o : out std_logic_vector(9 downto 0);
		entry_payload_9_o : out std_logic_vector(9 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(9 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_sta is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(9 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_11_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal TEMP_14_double_in_0 : std_logic_vector(19 downto 0);
	signal TEMP_14_double_out_0 : std_logic_vector(19 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "0000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "0000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "0000000000";
	entry_payload_2_o <= TEMP_3_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "0000000000";
	entry_payload_3_o <= TEMP_4_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "0000000000";
	entry_payload_4_o <= TEMP_5_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "0000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "0000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "0000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "0000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "0000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_11_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_11_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_11_res_2 <= entry_waiting_for_port_2;
	TEMP_11_res_3 <= entry_waiting_for_port_3;
	TEMP_11_res_4 <= entry_waiting_for_port_4;
	TEMP_11_res_5 <= entry_waiting_for_port_5;
	TEMP_11_res_6 <= entry_waiting_for_port_6;
	TEMP_11_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_4;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_5;
	TEMP_12_res_2 <= TEMP_11_res_2 or TEMP_11_res_6;
	TEMP_12_res_3 <= TEMP_11_res_3 or TEMP_11_res_7;
	-- Layer End
	TEMP_13_res_0 <= TEMP_12_res_0 or TEMP_12_res_2;
	TEMP_13_res_1 <= TEMP_12_res_1 or TEMP_12_res_3;
	-- Layer End
	port_ready_vec <= TEMP_13_res_0 or TEMP_13_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_14_double_in_0(0) <= entry_port_options_0(0);
	TEMP_14_double_in_0(10) <= entry_port_options_0(0);
	TEMP_14_double_in_0(1) <= entry_port_options_1(0);
	TEMP_14_double_in_0(11) <= entry_port_options_1(0);
	TEMP_14_double_in_0(2) <= entry_port_options_2(0);
	TEMP_14_double_in_0(12) <= entry_port_options_2(0);
	TEMP_14_double_in_0(3) <= entry_port_options_3(0);
	TEMP_14_double_in_0(13) <= entry_port_options_3(0);
	TEMP_14_double_in_0(4) <= entry_port_options_4(0);
	TEMP_14_double_in_0(14) <= entry_port_options_4(0);
	TEMP_14_double_in_0(5) <= entry_port_options_5(0);
	TEMP_14_double_in_0(15) <= entry_port_options_5(0);
	TEMP_14_double_in_0(6) <= entry_port_options_6(0);
	TEMP_14_double_in_0(16) <= entry_port_options_6(0);
	TEMP_14_double_in_0(7) <= entry_port_options_7(0);
	TEMP_14_double_in_0(17) <= entry_port_options_7(0);
	TEMP_14_double_in_0(8) <= entry_port_options_8(0);
	TEMP_14_double_in_0(18) <= entry_port_options_8(0);
	TEMP_14_double_in_0(9) <= entry_port_options_9(0);
	TEMP_14_double_in_0(19) <= entry_port_options_9(0);
	TEMP_14_double_out_0 <= TEMP_14_double_in_0 and not std_logic_vector( unsigned( TEMP_14_double_in_0 ) - unsigned( "0000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_14_double_out_0(0) or TEMP_14_double_out_0(10);
	entry_port_transfer_1(0) <= TEMP_14_double_out_0(1) or TEMP_14_double_out_0(11);
	entry_port_transfer_2(0) <= TEMP_14_double_out_0(2) or TEMP_14_double_out_0(12);
	entry_port_transfer_3(0) <= TEMP_14_double_out_0(3) or TEMP_14_double_out_0(13);
	entry_port_transfer_4(0) <= TEMP_14_double_out_0(4) or TEMP_14_double_out_0(14);
	entry_port_transfer_5(0) <= TEMP_14_double_out_0(5) or TEMP_14_double_out_0(15);
	entry_port_transfer_6(0) <= TEMP_14_double_out_0(6) or TEMP_14_double_out_0(16);
	entry_port_transfer_7(0) <= TEMP_14_double_out_0(7) or TEMP_14_double_out_0(17);
	entry_port_transfer_8(0) <= TEMP_14_double_out_0(8) or TEMP_14_double_out_0(18);
	entry_port_transfer_9(0) <= TEMP_14_double_out_0(9) or TEMP_14_double_out_0(19);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End


end architecture;


library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core_std is
	port(
		rst : in std_logic;
		clk : in std_logic;
		port_payload_0_i : in std_logic_vector(31 downto 0);
		port_valid_0_i : in std_logic;
		port_ready_0_o : out std_logic;
		entry_alloc_0_i : in std_logic;
		entry_alloc_1_i : in std_logic;
		entry_alloc_2_i : in std_logic;
		entry_alloc_3_i : in std_logic;
		entry_alloc_4_i : in std_logic;
		entry_alloc_5_i : in std_logic;
		entry_alloc_6_i : in std_logic;
		entry_alloc_7_i : in std_logic;
		entry_alloc_8_i : in std_logic;
		entry_alloc_9_i : in std_logic;
		entry_payload_valid_0_i : in std_logic;
		entry_payload_valid_1_i : in std_logic;
		entry_payload_valid_2_i : in std_logic;
		entry_payload_valid_3_i : in std_logic;
		entry_payload_valid_4_i : in std_logic;
		entry_payload_valid_5_i : in std_logic;
		entry_payload_valid_6_i : in std_logic;
		entry_payload_valid_7_i : in std_logic;
		entry_payload_valid_8_i : in std_logic;
		entry_payload_valid_9_i : in std_logic;
		entry_payload_0_o : out std_logic_vector(31 downto 0);
		entry_payload_1_o : out std_logic_vector(31 downto 0);
		entry_payload_2_o : out std_logic_vector(31 downto 0);
		entry_payload_3_o : out std_logic_vector(31 downto 0);
		entry_payload_4_o : out std_logic_vector(31 downto 0);
		entry_payload_5_o : out std_logic_vector(31 downto 0);
		entry_payload_6_o : out std_logic_vector(31 downto 0);
		entry_payload_7_o : out std_logic_vector(31 downto 0);
		entry_payload_8_o : out std_logic_vector(31 downto 0);
		entry_payload_9_o : out std_logic_vector(31 downto 0);
		entry_wen_0_o : out std_logic;
		entry_wen_1_o : out std_logic;
		entry_wen_2_o : out std_logic;
		entry_wen_3_o : out std_logic;
		entry_wen_4_o : out std_logic;
		entry_wen_5_o : out std_logic;
		entry_wen_6_o : out std_logic;
		entry_wen_7_o : out std_logic;
		entry_wen_8_o : out std_logic;
		entry_wen_9_o : out std_logic;
		queue_head_oh_i : in std_logic_vector(9 downto 0)
	);
end entity;

architecture arch of handshake_lsq_lsq1_core_std is
	signal entry_port_idx_oh_0 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_1 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_2 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_3 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_4 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_5 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_6 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_7 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_8 : std_logic_vector(0 downto 0);
	signal entry_port_idx_oh_9 : std_logic_vector(0 downto 0);
	signal TEMP_1_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_2_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_3_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_4_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_5_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_6_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_7_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_8_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_9_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_10_mux_0 : std_logic_vector(31 downto 0);
	signal entry_ptq_ready_0 : std_logic;
	signal entry_ptq_ready_1 : std_logic;
	signal entry_ptq_ready_2 : std_logic;
	signal entry_ptq_ready_3 : std_logic;
	signal entry_ptq_ready_4 : std_logic;
	signal entry_ptq_ready_5 : std_logic;
	signal entry_ptq_ready_6 : std_logic;
	signal entry_ptq_ready_7 : std_logic;
	signal entry_ptq_ready_8 : std_logic;
	signal entry_ptq_ready_9 : std_logic;
	signal entry_waiting_for_port_0 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_1 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_2 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_3 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_4 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_5 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_6 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_7 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_8 : std_logic_vector(0 downto 0);
	signal entry_waiting_for_port_9 : std_logic_vector(0 downto 0);
	signal port_ready_vec : std_logic_vector(0 downto 0);
	signal TEMP_11_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_4 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_5 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_6 : std_logic_vector(0 downto 0);
	signal TEMP_11_res_7 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_1 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_2 : std_logic_vector(0 downto 0);
	signal TEMP_12_res_3 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_0 : std_logic_vector(0 downto 0);
	signal TEMP_13_res_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_0 : std_logic_vector(0 downto 0);
	signal entry_port_options_1 : std_logic_vector(0 downto 0);
	signal entry_port_options_2 : std_logic_vector(0 downto 0);
	signal entry_port_options_3 : std_logic_vector(0 downto 0);
	signal entry_port_options_4 : std_logic_vector(0 downto 0);
	signal entry_port_options_5 : std_logic_vector(0 downto 0);
	signal entry_port_options_6 : std_logic_vector(0 downto 0);
	signal entry_port_options_7 : std_logic_vector(0 downto 0);
	signal entry_port_options_8 : std_logic_vector(0 downto 0);
	signal entry_port_options_9 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_0 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_1 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_2 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_3 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_4 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_5 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_6 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_7 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_8 : std_logic_vector(0 downto 0);
	signal entry_port_transfer_9 : std_logic_vector(0 downto 0);
	signal TEMP_14_double_in_0 : std_logic_vector(19 downto 0);
	signal TEMP_14_double_out_0 : std_logic_vector(19 downto 0);
begin
	entry_port_idx_oh_0 <= "1";
	entry_port_idx_oh_1 <= "1";
	entry_port_idx_oh_2 <= "1";
	entry_port_idx_oh_3 <= "1";
	entry_port_idx_oh_4 <= "1";
	entry_port_idx_oh_5 <= "1";
	entry_port_idx_oh_6 <= "1";
	entry_port_idx_oh_7 <= "1";
	entry_port_idx_oh_8 <= "1";
	entry_port_idx_oh_9 <= "1";
	-- Mux1H Begin
	-- Mux1H(entry_payload_0, port_payload, entry_port_idx_oh_0)
	TEMP_1_mux_0 <= port_payload_0_i when entry_port_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	entry_payload_0_o <= TEMP_1_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_1, port_payload, entry_port_idx_oh_1)
	TEMP_2_mux_0 <= port_payload_0_i when entry_port_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	entry_payload_1_o <= TEMP_2_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_2, port_payload, entry_port_idx_oh_2)
	TEMP_3_mux_0 <= port_payload_0_i when entry_port_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	entry_payload_2_o <= TEMP_3_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_3, port_payload, entry_port_idx_oh_3)
	TEMP_4_mux_0 <= port_payload_0_i when entry_port_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	entry_payload_3_o <= TEMP_4_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_4, port_payload, entry_port_idx_oh_4)
	TEMP_5_mux_0 <= port_payload_0_i when entry_port_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	entry_payload_4_o <= TEMP_5_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_5, port_payload, entry_port_idx_oh_5)
	TEMP_6_mux_0 <= port_payload_0_i when entry_port_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	entry_payload_5_o <= TEMP_6_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_6, port_payload, entry_port_idx_oh_6)
	TEMP_7_mux_0 <= port_payload_0_i when entry_port_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	entry_payload_6_o <= TEMP_7_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_7, port_payload, entry_port_idx_oh_7)
	TEMP_8_mux_0 <= port_payload_0_i when entry_port_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	entry_payload_7_o <= TEMP_8_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_8, port_payload, entry_port_idx_oh_8)
	TEMP_9_mux_0 <= port_payload_0_i when entry_port_idx_oh_8(0) = '1' else "00000000000000000000000000000000";
	entry_payload_8_o <= TEMP_9_mux_0;
	-- Mux1H End

	-- Mux1H Begin
	-- Mux1H(entry_payload_9, port_payload, entry_port_idx_oh_9)
	TEMP_10_mux_0 <= port_payload_0_i when entry_port_idx_oh_9(0) = '1' else "00000000000000000000000000000000";
	entry_payload_9_o <= TEMP_10_mux_0;
	-- Mux1H End

	entry_ptq_ready_0 <= entry_alloc_0_i and not entry_payload_valid_0_i;
	entry_ptq_ready_1 <= entry_alloc_1_i and not entry_payload_valid_1_i;
	entry_ptq_ready_2 <= entry_alloc_2_i and not entry_payload_valid_2_i;
	entry_ptq_ready_3 <= entry_alloc_3_i and not entry_payload_valid_3_i;
	entry_ptq_ready_4 <= entry_alloc_4_i and not entry_payload_valid_4_i;
	entry_ptq_ready_5 <= entry_alloc_5_i and not entry_payload_valid_5_i;
	entry_ptq_ready_6 <= entry_alloc_6_i and not entry_payload_valid_6_i;
	entry_ptq_ready_7 <= entry_alloc_7_i and not entry_payload_valid_7_i;
	entry_ptq_ready_8 <= entry_alloc_8_i and not entry_payload_valid_8_i;
	entry_ptq_ready_9 <= entry_alloc_9_i and not entry_payload_valid_9_i;
	entry_waiting_for_port_0 <= entry_port_idx_oh_0 when entry_ptq_ready_0 else "0";
	entry_waiting_for_port_1 <= entry_port_idx_oh_1 when entry_ptq_ready_1 else "0";
	entry_waiting_for_port_2 <= entry_port_idx_oh_2 when entry_ptq_ready_2 else "0";
	entry_waiting_for_port_3 <= entry_port_idx_oh_3 when entry_ptq_ready_3 else "0";
	entry_waiting_for_port_4 <= entry_port_idx_oh_4 when entry_ptq_ready_4 else "0";
	entry_waiting_for_port_5 <= entry_port_idx_oh_5 when entry_ptq_ready_5 else "0";
	entry_waiting_for_port_6 <= entry_port_idx_oh_6 when entry_ptq_ready_6 else "0";
	entry_waiting_for_port_7 <= entry_port_idx_oh_7 when entry_ptq_ready_7 else "0";
	entry_waiting_for_port_8 <= entry_port_idx_oh_8 when entry_ptq_ready_8 else "0";
	entry_waiting_for_port_9 <= entry_port_idx_oh_9 when entry_ptq_ready_9 else "0";
	-- Reduction Begin
	-- Reduce(port_ready_vec, entry_waiting_for_port, or)
	TEMP_11_res_0 <= entry_waiting_for_port_0 or entry_waiting_for_port_8;
	TEMP_11_res_1 <= entry_waiting_for_port_1 or entry_waiting_for_port_9;
	TEMP_11_res_2 <= entry_waiting_for_port_2;
	TEMP_11_res_3 <= entry_waiting_for_port_3;
	TEMP_11_res_4 <= entry_waiting_for_port_4;
	TEMP_11_res_5 <= entry_waiting_for_port_5;
	TEMP_11_res_6 <= entry_waiting_for_port_6;
	TEMP_11_res_7 <= entry_waiting_for_port_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_4;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_5;
	TEMP_12_res_2 <= TEMP_11_res_2 or TEMP_11_res_6;
	TEMP_12_res_3 <= TEMP_11_res_3 or TEMP_11_res_7;
	-- Layer End
	TEMP_13_res_0 <= TEMP_12_res_0 or TEMP_12_res_2;
	TEMP_13_res_1 <= TEMP_12_res_1 or TEMP_12_res_3;
	-- Layer End
	port_ready_vec <= TEMP_13_res_0 or TEMP_13_res_1;
	-- Reduction End

	port_ready_0_o <= port_ready_vec(0);
	entry_port_options_0(0) <= entry_waiting_for_port_0(0) and port_valid_0_i;
	entry_port_options_1(0) <= entry_waiting_for_port_1(0) and port_valid_0_i;
	entry_port_options_2(0) <= entry_waiting_for_port_2(0) and port_valid_0_i;
	entry_port_options_3(0) <= entry_waiting_for_port_3(0) and port_valid_0_i;
	entry_port_options_4(0) <= entry_waiting_for_port_4(0) and port_valid_0_i;
	entry_port_options_5(0) <= entry_waiting_for_port_5(0) and port_valid_0_i;
	entry_port_options_6(0) <= entry_waiting_for_port_6(0) and port_valid_0_i;
	entry_port_options_7(0) <= entry_waiting_for_port_7(0) and port_valid_0_i;
	entry_port_options_8(0) <= entry_waiting_for_port_8(0) and port_valid_0_i;
	entry_port_options_9(0) <= entry_waiting_for_port_9(0) and port_valid_0_i;
	-- Priority Masking Begin
	-- CyclicPriorityMask(entry_port_transfer, entry_port_options, queue_head_oh)
	TEMP_14_double_in_0(0) <= entry_port_options_0(0);
	TEMP_14_double_in_0(10) <= entry_port_options_0(0);
	TEMP_14_double_in_0(1) <= entry_port_options_1(0);
	TEMP_14_double_in_0(11) <= entry_port_options_1(0);
	TEMP_14_double_in_0(2) <= entry_port_options_2(0);
	TEMP_14_double_in_0(12) <= entry_port_options_2(0);
	TEMP_14_double_in_0(3) <= entry_port_options_3(0);
	TEMP_14_double_in_0(13) <= entry_port_options_3(0);
	TEMP_14_double_in_0(4) <= entry_port_options_4(0);
	TEMP_14_double_in_0(14) <= entry_port_options_4(0);
	TEMP_14_double_in_0(5) <= entry_port_options_5(0);
	TEMP_14_double_in_0(15) <= entry_port_options_5(0);
	TEMP_14_double_in_0(6) <= entry_port_options_6(0);
	TEMP_14_double_in_0(16) <= entry_port_options_6(0);
	TEMP_14_double_in_0(7) <= entry_port_options_7(0);
	TEMP_14_double_in_0(17) <= entry_port_options_7(0);
	TEMP_14_double_in_0(8) <= entry_port_options_8(0);
	TEMP_14_double_in_0(18) <= entry_port_options_8(0);
	TEMP_14_double_in_0(9) <= entry_port_options_9(0);
	TEMP_14_double_in_0(19) <= entry_port_options_9(0);
	TEMP_14_double_out_0 <= TEMP_14_double_in_0 and not std_logic_vector( unsigned( TEMP_14_double_in_0 ) - unsigned( "0000000000" & queue_head_oh_i ) );
	entry_port_transfer_0(0) <= TEMP_14_double_out_0(0) or TEMP_14_double_out_0(10);
	entry_port_transfer_1(0) <= TEMP_14_double_out_0(1) or TEMP_14_double_out_0(11);
	entry_port_transfer_2(0) <= TEMP_14_double_out_0(2) or TEMP_14_double_out_0(12);
	entry_port_transfer_3(0) <= TEMP_14_double_out_0(3) or TEMP_14_double_out_0(13);
	entry_port_transfer_4(0) <= TEMP_14_double_out_0(4) or TEMP_14_double_out_0(14);
	entry_port_transfer_5(0) <= TEMP_14_double_out_0(5) or TEMP_14_double_out_0(15);
	entry_port_transfer_6(0) <= TEMP_14_double_out_0(6) or TEMP_14_double_out_0(16);
	entry_port_transfer_7(0) <= TEMP_14_double_out_0(7) or TEMP_14_double_out_0(17);
	entry_port_transfer_8(0) <= TEMP_14_double_out_0(8) or TEMP_14_double_out_0(18);
	entry_port_transfer_9(0) <= TEMP_14_double_out_0(9) or TEMP_14_double_out_0(19);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(entry_wen_0, entry_port_transfer_0, or)
	entry_wen_0_o <= entry_port_transfer_0(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_1, entry_port_transfer_1, or)
	entry_wen_1_o <= entry_port_transfer_1(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_2, entry_port_transfer_2, or)
	entry_wen_2_o <= entry_port_transfer_2(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_3, entry_port_transfer_3, or)
	entry_wen_3_o <= entry_port_transfer_3(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_4, entry_port_transfer_4, or)
	entry_wen_4_o <= entry_port_transfer_4(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_5, entry_port_transfer_5, or)
	entry_wen_5_o <= entry_port_transfer_5(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_6, entry_port_transfer_6, or)
	entry_wen_6_o <= entry_port_transfer_6(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_7, entry_port_transfer_7, or)
	entry_wen_7_o <= entry_port_transfer_7(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_8, entry_port_transfer_8, or)
	entry_wen_8_o <= entry_port_transfer_8(0);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(entry_wen_9, entry_port_transfer_9, or)
	entry_wen_9_o <= entry_port_transfer_9(0);
	-- Reduction End


end architecture;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity handshake_lsq_lsq1_core is
	port(
		rst : in std_logic;
		clk : in std_logic;
		group_init_valid_0_i : in std_logic;
		group_init_ready_0_o : out std_logic;
		ldp_addr_0_i : in std_logic_vector(9 downto 0);
		ldp_addr_valid_0_i : in std_logic;
		ldp_addr_ready_0_o : out std_logic;
		ldp_data_0_o : out std_logic_vector(31 downto 0);
		ldp_data_valid_0_o : out std_logic;
		ldp_data_ready_0_i : in std_logic;
		stp_addr_0_i : in std_logic_vector(9 downto 0);
		stp_addr_valid_0_i : in std_logic;
		stp_addr_ready_0_o : out std_logic;
		stp_data_0_i : in std_logic_vector(31 downto 0);
		stp_data_valid_0_i : in std_logic;
		stp_data_ready_0_o : out std_logic;
		empty_o : out std_logic;
		rreq_valid_0_o : out std_logic;
		rreq_ready_0_i : in std_logic;
		rreq_id_0_o : out std_logic_vector(3 downto 0);
		rreq_addr_0_o : out std_logic_vector(9 downto 0);
		rresp_valid_0_i : in std_logic;
		rresp_ready_0_o : out std_logic;
		rresp_id_0_i : in std_logic_vector(3 downto 0);
		rresp_data_0_i : in std_logic_vector(31 downto 0);
		wreq_valid_0_o : out std_logic;
		wreq_ready_0_i : in std_logic;
		wreq_id_0_o : out std_logic_vector(3 downto 0);
		wreq_addr_0_o : out std_logic_vector(9 downto 0);
		wreq_data_0_o : out std_logic_vector(31 downto 0);
		wresp_valid_0_i : in std_logic;
		wresp_ready_0_o : out std_logic;
		wresp_id_0_i : in std_logic_vector(3 downto 0);
		memStart_ready_o : out std_logic;
		memStart_valid_i : in std_logic;
		ctrlEnd_ready_o : out std_logic;
		ctrlEnd_valid_i : in std_logic;
		memEnd_ready_i : in std_logic;
		memEnd_valid_o : out std_logic
	);
end entity;

architecture arch of handshake_lsq_lsq1_core is
	signal memStartReady : std_logic;
	signal memEndValid : std_logic;
	signal ctrlEndReady : std_logic;
	signal TEMP_GEN_MEM : std_logic;
	signal ldq_alloc_0_d : std_logic;
	signal ldq_alloc_0_q : std_logic;
	signal ldq_alloc_1_d : std_logic;
	signal ldq_alloc_1_q : std_logic;
	signal ldq_alloc_2_d : std_logic;
	signal ldq_alloc_2_q : std_logic;
	signal ldq_alloc_3_d : std_logic;
	signal ldq_alloc_3_q : std_logic;
	signal ldq_alloc_4_d : std_logic;
	signal ldq_alloc_4_q : std_logic;
	signal ldq_alloc_5_d : std_logic;
	signal ldq_alloc_5_q : std_logic;
	signal ldq_alloc_6_d : std_logic;
	signal ldq_alloc_6_q : std_logic;
	signal ldq_alloc_7_d : std_logic;
	signal ldq_alloc_7_q : std_logic;
	signal ldq_alloc_8_d : std_logic;
	signal ldq_alloc_8_q : std_logic;
	signal ldq_alloc_9_d : std_logic;
	signal ldq_alloc_9_q : std_logic;
	signal ldq_issue_0_d : std_logic;
	signal ldq_issue_0_q : std_logic;
	signal ldq_issue_1_d : std_logic;
	signal ldq_issue_1_q : std_logic;
	signal ldq_issue_2_d : std_logic;
	signal ldq_issue_2_q : std_logic;
	signal ldq_issue_3_d : std_logic;
	signal ldq_issue_3_q : std_logic;
	signal ldq_issue_4_d : std_logic;
	signal ldq_issue_4_q : std_logic;
	signal ldq_issue_5_d : std_logic;
	signal ldq_issue_5_q : std_logic;
	signal ldq_issue_6_d : std_logic;
	signal ldq_issue_6_q : std_logic;
	signal ldq_issue_7_d : std_logic;
	signal ldq_issue_7_q : std_logic;
	signal ldq_issue_8_d : std_logic;
	signal ldq_issue_8_q : std_logic;
	signal ldq_issue_9_d : std_logic;
	signal ldq_issue_9_q : std_logic;
	signal ldq_addr_valid_0_d : std_logic;
	signal ldq_addr_valid_0_q : std_logic;
	signal ldq_addr_valid_1_d : std_logic;
	signal ldq_addr_valid_1_q : std_logic;
	signal ldq_addr_valid_2_d : std_logic;
	signal ldq_addr_valid_2_q : std_logic;
	signal ldq_addr_valid_3_d : std_logic;
	signal ldq_addr_valid_3_q : std_logic;
	signal ldq_addr_valid_4_d : std_logic;
	signal ldq_addr_valid_4_q : std_logic;
	signal ldq_addr_valid_5_d : std_logic;
	signal ldq_addr_valid_5_q : std_logic;
	signal ldq_addr_valid_6_d : std_logic;
	signal ldq_addr_valid_6_q : std_logic;
	signal ldq_addr_valid_7_d : std_logic;
	signal ldq_addr_valid_7_q : std_logic;
	signal ldq_addr_valid_8_d : std_logic;
	signal ldq_addr_valid_8_q : std_logic;
	signal ldq_addr_valid_9_d : std_logic;
	signal ldq_addr_valid_9_q : std_logic;
	signal ldq_addr_0_d : std_logic_vector(9 downto 0);
	signal ldq_addr_0_q : std_logic_vector(9 downto 0);
	signal ldq_addr_1_d : std_logic_vector(9 downto 0);
	signal ldq_addr_1_q : std_logic_vector(9 downto 0);
	signal ldq_addr_2_d : std_logic_vector(9 downto 0);
	signal ldq_addr_2_q : std_logic_vector(9 downto 0);
	signal ldq_addr_3_d : std_logic_vector(9 downto 0);
	signal ldq_addr_3_q : std_logic_vector(9 downto 0);
	signal ldq_addr_4_d : std_logic_vector(9 downto 0);
	signal ldq_addr_4_q : std_logic_vector(9 downto 0);
	signal ldq_addr_5_d : std_logic_vector(9 downto 0);
	signal ldq_addr_5_q : std_logic_vector(9 downto 0);
	signal ldq_addr_6_d : std_logic_vector(9 downto 0);
	signal ldq_addr_6_q : std_logic_vector(9 downto 0);
	signal ldq_addr_7_d : std_logic_vector(9 downto 0);
	signal ldq_addr_7_q : std_logic_vector(9 downto 0);
	signal ldq_addr_8_d : std_logic_vector(9 downto 0);
	signal ldq_addr_8_q : std_logic_vector(9 downto 0);
	signal ldq_addr_9_d : std_logic_vector(9 downto 0);
	signal ldq_addr_9_q : std_logic_vector(9 downto 0);
	signal ldq_data_valid_0_d : std_logic;
	signal ldq_data_valid_0_q : std_logic;
	signal ldq_data_valid_1_d : std_logic;
	signal ldq_data_valid_1_q : std_logic;
	signal ldq_data_valid_2_d : std_logic;
	signal ldq_data_valid_2_q : std_logic;
	signal ldq_data_valid_3_d : std_logic;
	signal ldq_data_valid_3_q : std_logic;
	signal ldq_data_valid_4_d : std_logic;
	signal ldq_data_valid_4_q : std_logic;
	signal ldq_data_valid_5_d : std_logic;
	signal ldq_data_valid_5_q : std_logic;
	signal ldq_data_valid_6_d : std_logic;
	signal ldq_data_valid_6_q : std_logic;
	signal ldq_data_valid_7_d : std_logic;
	signal ldq_data_valid_7_q : std_logic;
	signal ldq_data_valid_8_d : std_logic;
	signal ldq_data_valid_8_q : std_logic;
	signal ldq_data_valid_9_d : std_logic;
	signal ldq_data_valid_9_q : std_logic;
	signal ldq_data_0_d : std_logic_vector(31 downto 0);
	signal ldq_data_0_q : std_logic_vector(31 downto 0);
	signal ldq_data_1_d : std_logic_vector(31 downto 0);
	signal ldq_data_1_q : std_logic_vector(31 downto 0);
	signal ldq_data_2_d : std_logic_vector(31 downto 0);
	signal ldq_data_2_q : std_logic_vector(31 downto 0);
	signal ldq_data_3_d : std_logic_vector(31 downto 0);
	signal ldq_data_3_q : std_logic_vector(31 downto 0);
	signal ldq_data_4_d : std_logic_vector(31 downto 0);
	signal ldq_data_4_q : std_logic_vector(31 downto 0);
	signal ldq_data_5_d : std_logic_vector(31 downto 0);
	signal ldq_data_5_q : std_logic_vector(31 downto 0);
	signal ldq_data_6_d : std_logic_vector(31 downto 0);
	signal ldq_data_6_q : std_logic_vector(31 downto 0);
	signal ldq_data_7_d : std_logic_vector(31 downto 0);
	signal ldq_data_7_q : std_logic_vector(31 downto 0);
	signal ldq_data_8_d : std_logic_vector(31 downto 0);
	signal ldq_data_8_q : std_logic_vector(31 downto 0);
	signal ldq_data_9_d : std_logic_vector(31 downto 0);
	signal ldq_data_9_q : std_logic_vector(31 downto 0);
	signal stq_alloc_0_d : std_logic;
	signal stq_alloc_0_q : std_logic;
	signal stq_alloc_1_d : std_logic;
	signal stq_alloc_1_q : std_logic;
	signal stq_alloc_2_d : std_logic;
	signal stq_alloc_2_q : std_logic;
	signal stq_alloc_3_d : std_logic;
	signal stq_alloc_3_q : std_logic;
	signal stq_alloc_4_d : std_logic;
	signal stq_alloc_4_q : std_logic;
	signal stq_alloc_5_d : std_logic;
	signal stq_alloc_5_q : std_logic;
	signal stq_alloc_6_d : std_logic;
	signal stq_alloc_6_q : std_logic;
	signal stq_alloc_7_d : std_logic;
	signal stq_alloc_7_q : std_logic;
	signal stq_alloc_8_d : std_logic;
	signal stq_alloc_8_q : std_logic;
	signal stq_alloc_9_d : std_logic;
	signal stq_alloc_9_q : std_logic;
	signal stq_addr_valid_0_d : std_logic;
	signal stq_addr_valid_0_q : std_logic;
	signal stq_addr_valid_1_d : std_logic;
	signal stq_addr_valid_1_q : std_logic;
	signal stq_addr_valid_2_d : std_logic;
	signal stq_addr_valid_2_q : std_logic;
	signal stq_addr_valid_3_d : std_logic;
	signal stq_addr_valid_3_q : std_logic;
	signal stq_addr_valid_4_d : std_logic;
	signal stq_addr_valid_4_q : std_logic;
	signal stq_addr_valid_5_d : std_logic;
	signal stq_addr_valid_5_q : std_logic;
	signal stq_addr_valid_6_d : std_logic;
	signal stq_addr_valid_6_q : std_logic;
	signal stq_addr_valid_7_d : std_logic;
	signal stq_addr_valid_7_q : std_logic;
	signal stq_addr_valid_8_d : std_logic;
	signal stq_addr_valid_8_q : std_logic;
	signal stq_addr_valid_9_d : std_logic;
	signal stq_addr_valid_9_q : std_logic;
	signal stq_addr_0_d : std_logic_vector(9 downto 0);
	signal stq_addr_0_q : std_logic_vector(9 downto 0);
	signal stq_addr_1_d : std_logic_vector(9 downto 0);
	signal stq_addr_1_q : std_logic_vector(9 downto 0);
	signal stq_addr_2_d : std_logic_vector(9 downto 0);
	signal stq_addr_2_q : std_logic_vector(9 downto 0);
	signal stq_addr_3_d : std_logic_vector(9 downto 0);
	signal stq_addr_3_q : std_logic_vector(9 downto 0);
	signal stq_addr_4_d : std_logic_vector(9 downto 0);
	signal stq_addr_4_q : std_logic_vector(9 downto 0);
	signal stq_addr_5_d : std_logic_vector(9 downto 0);
	signal stq_addr_5_q : std_logic_vector(9 downto 0);
	signal stq_addr_6_d : std_logic_vector(9 downto 0);
	signal stq_addr_6_q : std_logic_vector(9 downto 0);
	signal stq_addr_7_d : std_logic_vector(9 downto 0);
	signal stq_addr_7_q : std_logic_vector(9 downto 0);
	signal stq_addr_8_d : std_logic_vector(9 downto 0);
	signal stq_addr_8_q : std_logic_vector(9 downto 0);
	signal stq_addr_9_d : std_logic_vector(9 downto 0);
	signal stq_addr_9_q : std_logic_vector(9 downto 0);
	signal stq_data_valid_0_d : std_logic;
	signal stq_data_valid_0_q : std_logic;
	signal stq_data_valid_1_d : std_logic;
	signal stq_data_valid_1_q : std_logic;
	signal stq_data_valid_2_d : std_logic;
	signal stq_data_valid_2_q : std_logic;
	signal stq_data_valid_3_d : std_logic;
	signal stq_data_valid_3_q : std_logic;
	signal stq_data_valid_4_d : std_logic;
	signal stq_data_valid_4_q : std_logic;
	signal stq_data_valid_5_d : std_logic;
	signal stq_data_valid_5_q : std_logic;
	signal stq_data_valid_6_d : std_logic;
	signal stq_data_valid_6_q : std_logic;
	signal stq_data_valid_7_d : std_logic;
	signal stq_data_valid_7_q : std_logic;
	signal stq_data_valid_8_d : std_logic;
	signal stq_data_valid_8_q : std_logic;
	signal stq_data_valid_9_d : std_logic;
	signal stq_data_valid_9_q : std_logic;
	signal stq_data_0_d : std_logic_vector(31 downto 0);
	signal stq_data_0_q : std_logic_vector(31 downto 0);
	signal stq_data_1_d : std_logic_vector(31 downto 0);
	signal stq_data_1_q : std_logic_vector(31 downto 0);
	signal stq_data_2_d : std_logic_vector(31 downto 0);
	signal stq_data_2_q : std_logic_vector(31 downto 0);
	signal stq_data_3_d : std_logic_vector(31 downto 0);
	signal stq_data_3_q : std_logic_vector(31 downto 0);
	signal stq_data_4_d : std_logic_vector(31 downto 0);
	signal stq_data_4_q : std_logic_vector(31 downto 0);
	signal stq_data_5_d : std_logic_vector(31 downto 0);
	signal stq_data_5_q : std_logic_vector(31 downto 0);
	signal stq_data_6_d : std_logic_vector(31 downto 0);
	signal stq_data_6_q : std_logic_vector(31 downto 0);
	signal stq_data_7_d : std_logic_vector(31 downto 0);
	signal stq_data_7_q : std_logic_vector(31 downto 0);
	signal stq_data_8_d : std_logic_vector(31 downto 0);
	signal stq_data_8_q : std_logic_vector(31 downto 0);
	signal stq_data_9_d : std_logic_vector(31 downto 0);
	signal stq_data_9_q : std_logic_vector(31 downto 0);
	signal store_is_older_0_d : std_logic_vector(9 downto 0);
	signal store_is_older_0_q : std_logic_vector(9 downto 0);
	signal store_is_older_1_d : std_logic_vector(9 downto 0);
	signal store_is_older_1_q : std_logic_vector(9 downto 0);
	signal store_is_older_2_d : std_logic_vector(9 downto 0);
	signal store_is_older_2_q : std_logic_vector(9 downto 0);
	signal store_is_older_3_d : std_logic_vector(9 downto 0);
	signal store_is_older_3_q : std_logic_vector(9 downto 0);
	signal store_is_older_4_d : std_logic_vector(9 downto 0);
	signal store_is_older_4_q : std_logic_vector(9 downto 0);
	signal store_is_older_5_d : std_logic_vector(9 downto 0);
	signal store_is_older_5_q : std_logic_vector(9 downto 0);
	signal store_is_older_6_d : std_logic_vector(9 downto 0);
	signal store_is_older_6_q : std_logic_vector(9 downto 0);
	signal store_is_older_7_d : std_logic_vector(9 downto 0);
	signal store_is_older_7_q : std_logic_vector(9 downto 0);
	signal store_is_older_8_d : std_logic_vector(9 downto 0);
	signal store_is_older_8_q : std_logic_vector(9 downto 0);
	signal store_is_older_9_d : std_logic_vector(9 downto 0);
	signal store_is_older_9_q : std_logic_vector(9 downto 0);
	signal ldq_tail_d : std_logic_vector(3 downto 0);
	signal ldq_tail_q : std_logic_vector(3 downto 0);
	signal ldq_head_d : std_logic_vector(3 downto 0);
	signal ldq_head_q : std_logic_vector(3 downto 0);
	signal stq_tail_d : std_logic_vector(3 downto 0);
	signal stq_tail_q : std_logic_vector(3 downto 0);
	signal stq_head_d : std_logic_vector(3 downto 0);
	signal stq_head_q : std_logic_vector(3 downto 0);
	signal stq_issue_d : std_logic_vector(3 downto 0);
	signal stq_issue_q : std_logic_vector(3 downto 0);
	signal stq_resp_d : std_logic_vector(3 downto 0);
	signal stq_resp_q : std_logic_vector(3 downto 0);
	signal ldq_wen_0 : std_logic;
	signal ldq_wen_1 : std_logic;
	signal ldq_wen_2 : std_logic;
	signal ldq_wen_3 : std_logic;
	signal ldq_wen_4 : std_logic;
	signal ldq_wen_5 : std_logic;
	signal ldq_wen_6 : std_logic;
	signal ldq_wen_7 : std_logic;
	signal ldq_wen_8 : std_logic;
	signal ldq_wen_9 : std_logic;
	signal ldq_addr_wen_0 : std_logic;
	signal ldq_addr_wen_1 : std_logic;
	signal ldq_addr_wen_2 : std_logic;
	signal ldq_addr_wen_3 : std_logic;
	signal ldq_addr_wen_4 : std_logic;
	signal ldq_addr_wen_5 : std_logic;
	signal ldq_addr_wen_6 : std_logic;
	signal ldq_addr_wen_7 : std_logic;
	signal ldq_addr_wen_8 : std_logic;
	signal ldq_addr_wen_9 : std_logic;
	signal ldq_reset_0 : std_logic;
	signal ldq_reset_1 : std_logic;
	signal ldq_reset_2 : std_logic;
	signal ldq_reset_3 : std_logic;
	signal ldq_reset_4 : std_logic;
	signal ldq_reset_5 : std_logic;
	signal ldq_reset_6 : std_logic;
	signal ldq_reset_7 : std_logic;
	signal ldq_reset_8 : std_logic;
	signal ldq_reset_9 : std_logic;
	signal stq_wen_0 : std_logic;
	signal stq_wen_1 : std_logic;
	signal stq_wen_2 : std_logic;
	signal stq_wen_3 : std_logic;
	signal stq_wen_4 : std_logic;
	signal stq_wen_5 : std_logic;
	signal stq_wen_6 : std_logic;
	signal stq_wen_7 : std_logic;
	signal stq_wen_8 : std_logic;
	signal stq_wen_9 : std_logic;
	signal stq_addr_wen_0 : std_logic;
	signal stq_addr_wen_1 : std_logic;
	signal stq_addr_wen_2 : std_logic;
	signal stq_addr_wen_3 : std_logic;
	signal stq_addr_wen_4 : std_logic;
	signal stq_addr_wen_5 : std_logic;
	signal stq_addr_wen_6 : std_logic;
	signal stq_addr_wen_7 : std_logic;
	signal stq_addr_wen_8 : std_logic;
	signal stq_addr_wen_9 : std_logic;
	signal stq_data_wen_0 : std_logic;
	signal stq_data_wen_1 : std_logic;
	signal stq_data_wen_2 : std_logic;
	signal stq_data_wen_3 : std_logic;
	signal stq_data_wen_4 : std_logic;
	signal stq_data_wen_5 : std_logic;
	signal stq_data_wen_6 : std_logic;
	signal stq_data_wen_7 : std_logic;
	signal stq_data_wen_8 : std_logic;
	signal stq_data_wen_9 : std_logic;
	signal stq_reset_0 : std_logic;
	signal stq_reset_1 : std_logic;
	signal stq_reset_2 : std_logic;
	signal stq_reset_3 : std_logic;
	signal stq_reset_4 : std_logic;
	signal stq_reset_5 : std_logic;
	signal stq_reset_6 : std_logic;
	signal stq_reset_7 : std_logic;
	signal stq_reset_8 : std_logic;
	signal stq_reset_9 : std_logic;
	signal ldq_data_wen_0 : std_logic;
	signal ldq_data_wen_1 : std_logic;
	signal ldq_data_wen_2 : std_logic;
	signal ldq_data_wen_3 : std_logic;
	signal ldq_data_wen_4 : std_logic;
	signal ldq_data_wen_5 : std_logic;
	signal ldq_data_wen_6 : std_logic;
	signal ldq_data_wen_7 : std_logic;
	signal ldq_data_wen_8 : std_logic;
	signal ldq_data_wen_9 : std_logic;
	signal ldq_issue_set_0 : std_logic;
	signal ldq_issue_set_1 : std_logic;
	signal ldq_issue_set_2 : std_logic;
	signal ldq_issue_set_3 : std_logic;
	signal ldq_issue_set_4 : std_logic;
	signal ldq_issue_set_5 : std_logic;
	signal ldq_issue_set_6 : std_logic;
	signal ldq_issue_set_7 : std_logic;
	signal ldq_issue_set_8 : std_logic;
	signal ldq_issue_set_9 : std_logic;
	signal ga_ls_order_0 : std_logic_vector(9 downto 0);
	signal ga_ls_order_1 : std_logic_vector(9 downto 0);
	signal ga_ls_order_2 : std_logic_vector(9 downto 0);
	signal ga_ls_order_3 : std_logic_vector(9 downto 0);
	signal ga_ls_order_4 : std_logic_vector(9 downto 0);
	signal ga_ls_order_5 : std_logic_vector(9 downto 0);
	signal ga_ls_order_6 : std_logic_vector(9 downto 0);
	signal ga_ls_order_7 : std_logic_vector(9 downto 0);
	signal ga_ls_order_8 : std_logic_vector(9 downto 0);
	signal ga_ls_order_9 : std_logic_vector(9 downto 0);
	signal num_loads : std_logic_vector(3 downto 0);
	signal num_stores : std_logic_vector(3 downto 0);
	signal stq_issue_en : std_logic;
	signal stq_resp_en : std_logic;
	signal ldq_empty : std_logic;
	signal stq_empty : std_logic;
	signal ldq_head_oh : std_logic_vector(9 downto 0);
	signal stq_head_oh : std_logic_vector(9 downto 0);
	signal ldq_alloc_next_0 : std_logic;
	signal ldq_alloc_next_1 : std_logic;
	signal ldq_alloc_next_2 : std_logic;
	signal ldq_alloc_next_3 : std_logic;
	signal ldq_alloc_next_4 : std_logic;
	signal ldq_alloc_next_5 : std_logic;
	signal ldq_alloc_next_6 : std_logic;
	signal ldq_alloc_next_7 : std_logic;
	signal ldq_alloc_next_8 : std_logic;
	signal ldq_alloc_next_9 : std_logic;
	signal stq_alloc_next_0 : std_logic;
	signal stq_alloc_next_1 : std_logic;
	signal stq_alloc_next_2 : std_logic;
	signal stq_alloc_next_3 : std_logic;
	signal stq_alloc_next_4 : std_logic;
	signal stq_alloc_next_5 : std_logic;
	signal stq_alloc_next_6 : std_logic;
	signal stq_alloc_next_7 : std_logic;
	signal stq_alloc_next_8 : std_logic;
	signal stq_alloc_next_9 : std_logic;
	signal ldq_not_empty : std_logic;
	signal stq_not_empty : std_logic;
	signal TEMP_1_res_0 : std_logic;
	signal TEMP_1_res_1 : std_logic;
	signal TEMP_1_res_2 : std_logic;
	signal TEMP_1_res_3 : std_logic;
	signal TEMP_1_res_4 : std_logic;
	signal TEMP_1_res_5 : std_logic;
	signal TEMP_1_res_6 : std_logic;
	signal TEMP_1_res_7 : std_logic;
	signal TEMP_2_res_0 : std_logic;
	signal TEMP_2_res_1 : std_logic;
	signal TEMP_2_res_2 : std_logic;
	signal TEMP_2_res_3 : std_logic;
	signal TEMP_3_res_0 : std_logic;
	signal TEMP_3_res_1 : std_logic;
	signal TEMP_4_sum : std_logic_vector(4 downto 0);
	signal TEMP_4_res : std_logic_vector(4 downto 0);
	signal TEMP_5_sum : std_logic_vector(4 downto 0);
	signal TEMP_5_res : std_logic_vector(4 downto 0);
	signal ldq_tail_oh : std_logic_vector(9 downto 0);
	signal ldq_head_next_oh : std_logic_vector(9 downto 0);
	signal ldq_head_next : std_logic_vector(3 downto 0);
	signal ldq_head_sel : std_logic;
	signal TEMP_6_double_in : std_logic_vector(19 downto 0);
	signal TEMP_6_double_out : std_logic_vector(19 downto 0);
	signal TEMP_7_res_0 : std_logic;
	signal TEMP_7_res_1 : std_logic;
	signal TEMP_7_res_2 : std_logic;
	signal TEMP_7_res_3 : std_logic;
	signal TEMP_7_res_4 : std_logic;
	signal TEMP_7_res_5 : std_logic;
	signal TEMP_7_res_6 : std_logic;
	signal TEMP_7_res_7 : std_logic;
	signal TEMP_8_res_0 : std_logic;
	signal TEMP_8_res_1 : std_logic;
	signal TEMP_8_res_2 : std_logic;
	signal TEMP_8_res_3 : std_logic;
	signal TEMP_9_res_0 : std_logic;
	signal TEMP_9_res_1 : std_logic;
	signal TEMP_10_in_0_0 : std_logic;
	signal TEMP_10_in_0_1 : std_logic;
	signal TEMP_10_in_0_2 : std_logic;
	signal TEMP_10_in_0_3 : std_logic;
	signal TEMP_10_in_0_4 : std_logic;
	signal TEMP_10_in_0_5 : std_logic;
	signal TEMP_10_in_0_6 : std_logic;
	signal TEMP_10_in_0_7 : std_logic;
	signal TEMP_10_in_0_8 : std_logic;
	signal TEMP_10_in_0_9 : std_logic;
	signal TEMP_10_out_0 : std_logic;
	signal TEMP_11_res_0 : std_logic;
	signal TEMP_11_res_1 : std_logic;
	signal TEMP_11_res_2 : std_logic;
	signal TEMP_11_res_3 : std_logic;
	signal TEMP_11_res_4 : std_logic;
	signal TEMP_11_res_5 : std_logic;
	signal TEMP_11_res_6 : std_logic;
	signal TEMP_11_res_7 : std_logic;
	signal TEMP_12_res_0 : std_logic;
	signal TEMP_12_res_1 : std_logic;
	signal TEMP_12_res_2 : std_logic;
	signal TEMP_12_res_3 : std_logic;
	signal TEMP_13_res_0 : std_logic;
	signal TEMP_13_res_1 : std_logic;
	signal TEMP_13_in_1_0 : std_logic;
	signal TEMP_13_in_1_1 : std_logic;
	signal TEMP_13_in_1_2 : std_logic;
	signal TEMP_13_in_1_3 : std_logic;
	signal TEMP_13_in_1_4 : std_logic;
	signal TEMP_13_in_1_5 : std_logic;
	signal TEMP_13_in_1_6 : std_logic;
	signal TEMP_13_in_1_7 : std_logic;
	signal TEMP_13_in_1_8 : std_logic;
	signal TEMP_13_in_1_9 : std_logic;
	signal TEMP_13_out_1 : std_logic;
	signal TEMP_14_res_0 : std_logic;
	signal TEMP_14_res_1 : std_logic;
	signal TEMP_14_res_2 : std_logic;
	signal TEMP_14_res_3 : std_logic;
	signal TEMP_14_res_4 : std_logic;
	signal TEMP_14_res_5 : std_logic;
	signal TEMP_14_res_6 : std_logic;
	signal TEMP_14_res_7 : std_logic;
	signal TEMP_15_res_0 : std_logic;
	signal TEMP_15_res_1 : std_logic;
	signal TEMP_15_res_2 : std_logic;
	signal TEMP_15_res_3 : std_logic;
	signal TEMP_16_res_0 : std_logic;
	signal TEMP_16_res_1 : std_logic;
	signal TEMP_16_in_2_0 : std_logic;
	signal TEMP_16_in_2_1 : std_logic;
	signal TEMP_16_in_2_2 : std_logic;
	signal TEMP_16_in_2_3 : std_logic;
	signal TEMP_16_in_2_4 : std_logic;
	signal TEMP_16_in_2_5 : std_logic;
	signal TEMP_16_in_2_6 : std_logic;
	signal TEMP_16_in_2_7 : std_logic;
	signal TEMP_16_in_2_8 : std_logic;
	signal TEMP_16_in_2_9 : std_logic;
	signal TEMP_16_out_2 : std_logic;
	signal TEMP_17_res_0 : std_logic;
	signal TEMP_17_res_1 : std_logic;
	signal TEMP_17_res_2 : std_logic;
	signal TEMP_17_res_3 : std_logic;
	signal TEMP_17_res_4 : std_logic;
	signal TEMP_17_res_5 : std_logic;
	signal TEMP_17_res_6 : std_logic;
	signal TEMP_17_res_7 : std_logic;
	signal TEMP_18_res_0 : std_logic;
	signal TEMP_18_res_1 : std_logic;
	signal TEMP_18_res_2 : std_logic;
	signal TEMP_18_res_3 : std_logic;
	signal TEMP_19_res_0 : std_logic;
	signal TEMP_19_res_1 : std_logic;
	signal TEMP_19_in_3_0 : std_logic;
	signal TEMP_19_in_3_1 : std_logic;
	signal TEMP_19_in_3_2 : std_logic;
	signal TEMP_19_in_3_3 : std_logic;
	signal TEMP_19_in_3_4 : std_logic;
	signal TEMP_19_in_3_5 : std_logic;
	signal TEMP_19_in_3_6 : std_logic;
	signal TEMP_19_in_3_7 : std_logic;
	signal TEMP_19_in_3_8 : std_logic;
	signal TEMP_19_in_3_9 : std_logic;
	signal TEMP_19_out_3 : std_logic;
	signal TEMP_20_res_0 : std_logic;
	signal TEMP_20_res_1 : std_logic;
	signal TEMP_20_res_2 : std_logic;
	signal TEMP_20_res_3 : std_logic;
	signal TEMP_20_res_4 : std_logic;
	signal TEMP_20_res_5 : std_logic;
	signal TEMP_20_res_6 : std_logic;
	signal TEMP_20_res_7 : std_logic;
	signal TEMP_21_res_0 : std_logic;
	signal TEMP_21_res_1 : std_logic;
	signal TEMP_21_res_2 : std_logic;
	signal TEMP_21_res_3 : std_logic;
	signal TEMP_22_res_0 : std_logic;
	signal TEMP_22_res_1 : std_logic;
	signal stq_tail_oh : std_logic_vector(9 downto 0);
	signal stq_head_next_oh : std_logic_vector(9 downto 0);
	signal stq_head_next : std_logic_vector(3 downto 0);
	signal stq_head_sel : std_logic;
	signal load_idx_oh_0 : std_logic_vector(9 downto 0);
	signal load_en_0 : std_logic;
	signal store_idx : std_logic_vector(3 downto 0);
	signal store_en : std_logic;
	signal bypass_idx_oh_0 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_1 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_2 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_3 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_4 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_5 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_6 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_7 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_8 : std_logic_vector(9 downto 0);
	signal bypass_idx_oh_9 : std_logic_vector(9 downto 0);
	signal bypass_en_0 : std_logic;
	signal bypass_en_1 : std_logic;
	signal bypass_en_2 : std_logic;
	signal bypass_en_3 : std_logic;
	signal bypass_en_4 : std_logic;
	signal bypass_en_5 : std_logic;
	signal bypass_en_6 : std_logic;
	signal bypass_en_7 : std_logic;
	signal bypass_en_8 : std_logic;
	signal bypass_en_9 : std_logic;
	signal ld_st_conflict_0 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_1 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_2 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_3 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_4 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_5 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_6 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_7 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_8 : std_logic_vector(9 downto 0);
	signal ld_st_conflict_9 : std_logic_vector(9 downto 0);
	signal can_bypass_0 : std_logic_vector(9 downto 0);
	signal can_bypass_1 : std_logic_vector(9 downto 0);
	signal can_bypass_2 : std_logic_vector(9 downto 0);
	signal can_bypass_3 : std_logic_vector(9 downto 0);
	signal can_bypass_4 : std_logic_vector(9 downto 0);
	signal can_bypass_5 : std_logic_vector(9 downto 0);
	signal can_bypass_6 : std_logic_vector(9 downto 0);
	signal can_bypass_7 : std_logic_vector(9 downto 0);
	signal can_bypass_8 : std_logic_vector(9 downto 0);
	signal can_bypass_9 : std_logic_vector(9 downto 0);
	signal addr_valid_0 : std_logic_vector(9 downto 0);
	signal addr_valid_1 : std_logic_vector(9 downto 0);
	signal addr_valid_2 : std_logic_vector(9 downto 0);
	signal addr_valid_3 : std_logic_vector(9 downto 0);
	signal addr_valid_4 : std_logic_vector(9 downto 0);
	signal addr_valid_5 : std_logic_vector(9 downto 0);
	signal addr_valid_6 : std_logic_vector(9 downto 0);
	signal addr_valid_7 : std_logic_vector(9 downto 0);
	signal addr_valid_8 : std_logic_vector(9 downto 0);
	signal addr_valid_9 : std_logic_vector(9 downto 0);
	signal addr_same_0 : std_logic_vector(9 downto 0);
	signal addr_same_1 : std_logic_vector(9 downto 0);
	signal addr_same_2 : std_logic_vector(9 downto 0);
	signal addr_same_3 : std_logic_vector(9 downto 0);
	signal addr_same_4 : std_logic_vector(9 downto 0);
	signal addr_same_5 : std_logic_vector(9 downto 0);
	signal addr_same_6 : std_logic_vector(9 downto 0);
	signal addr_same_7 : std_logic_vector(9 downto 0);
	signal addr_same_8 : std_logic_vector(9 downto 0);
	signal addr_same_9 : std_logic_vector(9 downto 0);
	signal load_conflict_0 : std_logic;
	signal load_conflict_1 : std_logic;
	signal load_conflict_2 : std_logic;
	signal load_conflict_3 : std_logic;
	signal load_conflict_4 : std_logic;
	signal load_conflict_5 : std_logic;
	signal load_conflict_6 : std_logic;
	signal load_conflict_7 : std_logic;
	signal load_conflict_8 : std_logic;
	signal load_conflict_9 : std_logic;
	signal load_req_valid_0 : std_logic;
	signal load_req_valid_1 : std_logic;
	signal load_req_valid_2 : std_logic;
	signal load_req_valid_3 : std_logic;
	signal load_req_valid_4 : std_logic;
	signal load_req_valid_5 : std_logic;
	signal load_req_valid_6 : std_logic;
	signal load_req_valid_7 : std_logic;
	signal load_req_valid_8 : std_logic;
	signal load_req_valid_9 : std_logic;
	signal can_load_0 : std_logic;
	signal can_load_1 : std_logic;
	signal can_load_2 : std_logic;
	signal can_load_3 : std_logic;
	signal can_load_4 : std_logic;
	signal can_load_5 : std_logic;
	signal can_load_6 : std_logic;
	signal can_load_7 : std_logic;
	signal can_load_8 : std_logic;
	signal can_load_9 : std_logic;
	signal TEMP_23_res : std_logic_vector(7 downto 0);
	signal TEMP_24_res : std_logic_vector(3 downto 0);
	signal TEMP_25_res : std_logic_vector(1 downto 0);
	signal TEMP_26_res : std_logic_vector(7 downto 0);
	signal TEMP_27_res : std_logic_vector(3 downto 0);
	signal TEMP_28_res : std_logic_vector(1 downto 0);
	signal TEMP_29_res : std_logic_vector(7 downto 0);
	signal TEMP_30_res : std_logic_vector(3 downto 0);
	signal TEMP_31_res : std_logic_vector(1 downto 0);
	signal TEMP_32_res : std_logic_vector(7 downto 0);
	signal TEMP_33_res : std_logic_vector(3 downto 0);
	signal TEMP_34_res : std_logic_vector(1 downto 0);
	signal TEMP_35_res : std_logic_vector(7 downto 0);
	signal TEMP_36_res : std_logic_vector(3 downto 0);
	signal TEMP_37_res : std_logic_vector(1 downto 0);
	signal TEMP_38_res : std_logic_vector(7 downto 0);
	signal TEMP_39_res : std_logic_vector(3 downto 0);
	signal TEMP_40_res : std_logic_vector(1 downto 0);
	signal TEMP_41_res : std_logic_vector(7 downto 0);
	signal TEMP_42_res : std_logic_vector(3 downto 0);
	signal TEMP_43_res : std_logic_vector(1 downto 0);
	signal TEMP_44_res : std_logic_vector(7 downto 0);
	signal TEMP_45_res : std_logic_vector(3 downto 0);
	signal TEMP_46_res : std_logic_vector(1 downto 0);
	signal TEMP_47_res : std_logic_vector(7 downto 0);
	signal TEMP_48_res : std_logic_vector(3 downto 0);
	signal TEMP_49_res : std_logic_vector(1 downto 0);
	signal TEMP_50_res : std_logic_vector(7 downto 0);
	signal TEMP_51_res : std_logic_vector(3 downto 0);
	signal TEMP_52_res : std_logic_vector(1 downto 0);
	signal TEMP_53_double_in : std_logic_vector(19 downto 0);
	signal TEMP_53_double_out : std_logic_vector(19 downto 0);
	signal TEMP_54_res_0 : std_logic;
	signal TEMP_54_res_1 : std_logic;
	signal TEMP_54_res_2 : std_logic;
	signal TEMP_54_res_3 : std_logic;
	signal TEMP_54_res_4 : std_logic;
	signal TEMP_54_res_5 : std_logic;
	signal TEMP_54_res_6 : std_logic;
	signal TEMP_54_res_7 : std_logic;
	signal TEMP_55_res_0 : std_logic;
	signal TEMP_55_res_1 : std_logic;
	signal TEMP_55_res_2 : std_logic;
	signal TEMP_55_res_3 : std_logic;
	signal TEMP_56_res_0 : std_logic;
	signal TEMP_56_res_1 : std_logic;
	signal st_ld_conflict : std_logic_vector(9 downto 0);
	signal store_conflict : std_logic;
	signal store_valid : std_logic;
	signal store_data_valid : std_logic;
	signal store_addr_valid : std_logic;
	signal TEMP_57_res : std_logic_vector(7 downto 0);
	signal TEMP_58_res : std_logic_vector(3 downto 0);
	signal TEMP_59_res : std_logic_vector(1 downto 0);
	signal stq_last_oh : std_logic_vector(9 downto 0);
	signal bypass_en_vec_0 : std_logic_vector(9 downto 0);
	signal TEMP_60_double_in : std_logic_vector(19 downto 0);
	signal TEMP_60_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_60_double_out : std_logic_vector(19 downto 0);
	signal TEMP_61_res : std_logic_vector(7 downto 0);
	signal TEMP_62_res : std_logic_vector(3 downto 0);
	signal TEMP_63_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_1 : std_logic_vector(9 downto 0);
	signal TEMP_64_double_in : std_logic_vector(19 downto 0);
	signal TEMP_64_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_64_double_out : std_logic_vector(19 downto 0);
	signal TEMP_65_res : std_logic_vector(7 downto 0);
	signal TEMP_66_res : std_logic_vector(3 downto 0);
	signal TEMP_67_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_2 : std_logic_vector(9 downto 0);
	signal TEMP_68_double_in : std_logic_vector(19 downto 0);
	signal TEMP_68_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_68_double_out : std_logic_vector(19 downto 0);
	signal TEMP_69_res : std_logic_vector(7 downto 0);
	signal TEMP_70_res : std_logic_vector(3 downto 0);
	signal TEMP_71_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_3 : std_logic_vector(9 downto 0);
	signal TEMP_72_double_in : std_logic_vector(19 downto 0);
	signal TEMP_72_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_72_double_out : std_logic_vector(19 downto 0);
	signal TEMP_73_res : std_logic_vector(7 downto 0);
	signal TEMP_74_res : std_logic_vector(3 downto 0);
	signal TEMP_75_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_4 : std_logic_vector(9 downto 0);
	signal TEMP_76_double_in : std_logic_vector(19 downto 0);
	signal TEMP_76_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_76_double_out : std_logic_vector(19 downto 0);
	signal TEMP_77_res : std_logic_vector(7 downto 0);
	signal TEMP_78_res : std_logic_vector(3 downto 0);
	signal TEMP_79_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_5 : std_logic_vector(9 downto 0);
	signal TEMP_80_double_in : std_logic_vector(19 downto 0);
	signal TEMP_80_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_80_double_out : std_logic_vector(19 downto 0);
	signal TEMP_81_res : std_logic_vector(7 downto 0);
	signal TEMP_82_res : std_logic_vector(3 downto 0);
	signal TEMP_83_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_6 : std_logic_vector(9 downto 0);
	signal TEMP_84_double_in : std_logic_vector(19 downto 0);
	signal TEMP_84_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_84_double_out : std_logic_vector(19 downto 0);
	signal TEMP_85_res : std_logic_vector(7 downto 0);
	signal TEMP_86_res : std_logic_vector(3 downto 0);
	signal TEMP_87_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_7 : std_logic_vector(9 downto 0);
	signal TEMP_88_double_in : std_logic_vector(19 downto 0);
	signal TEMP_88_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_88_double_out : std_logic_vector(19 downto 0);
	signal TEMP_89_res : std_logic_vector(7 downto 0);
	signal TEMP_90_res : std_logic_vector(3 downto 0);
	signal TEMP_91_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_8 : std_logic_vector(9 downto 0);
	signal TEMP_92_double_in : std_logic_vector(19 downto 0);
	signal TEMP_92_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_92_double_out : std_logic_vector(19 downto 0);
	signal TEMP_93_res : std_logic_vector(7 downto 0);
	signal TEMP_94_res : std_logic_vector(3 downto 0);
	signal TEMP_95_res : std_logic_vector(1 downto 0);
	signal bypass_en_vec_9 : std_logic_vector(9 downto 0);
	signal TEMP_96_double_in : std_logic_vector(19 downto 0);
	signal TEMP_96_base_rev : std_logic_vector(9 downto 0);
	signal TEMP_96_double_out : std_logic_vector(19 downto 0);
	signal TEMP_97_res : std_logic_vector(7 downto 0);
	signal TEMP_98_res : std_logic_vector(3 downto 0);
	signal TEMP_99_res : std_logic_vector(1 downto 0);
	signal TEMP_100_in_0_0 : std_logic;
	signal TEMP_100_in_0_1 : std_logic;
	signal TEMP_100_in_0_2 : std_logic;
	signal TEMP_100_in_0_3 : std_logic;
	signal TEMP_100_in_0_4 : std_logic;
	signal TEMP_100_in_0_5 : std_logic;
	signal TEMP_100_in_0_6 : std_logic;
	signal TEMP_100_in_0_7 : std_logic;
	signal TEMP_100_in_0_8 : std_logic;
	signal TEMP_100_in_0_9 : std_logic;
	signal TEMP_100_out_0 : std_logic;
	signal TEMP_101_res_0 : std_logic;
	signal TEMP_101_res_1 : std_logic;
	signal TEMP_101_res_2 : std_logic;
	signal TEMP_101_res_3 : std_logic;
	signal TEMP_101_res_4 : std_logic;
	signal TEMP_101_res_5 : std_logic;
	signal TEMP_101_res_6 : std_logic;
	signal TEMP_101_res_7 : std_logic;
	signal TEMP_102_res_0 : std_logic;
	signal TEMP_102_res_1 : std_logic;
	signal TEMP_102_res_2 : std_logic;
	signal TEMP_102_res_3 : std_logic;
	signal TEMP_103_res_0 : std_logic;
	signal TEMP_103_res_1 : std_logic;
	signal TEMP_103_in_1_0 : std_logic;
	signal TEMP_103_in_1_1 : std_logic;
	signal TEMP_103_in_1_2 : std_logic;
	signal TEMP_103_in_1_3 : std_logic;
	signal TEMP_103_in_1_4 : std_logic;
	signal TEMP_103_in_1_5 : std_logic;
	signal TEMP_103_in_1_6 : std_logic;
	signal TEMP_103_in_1_7 : std_logic;
	signal TEMP_103_in_1_8 : std_logic;
	signal TEMP_103_in_1_9 : std_logic;
	signal TEMP_103_out_1 : std_logic;
	signal TEMP_104_res_0 : std_logic;
	signal TEMP_104_res_1 : std_logic;
	signal TEMP_104_res_2 : std_logic;
	signal TEMP_104_res_3 : std_logic;
	signal TEMP_104_res_4 : std_logic;
	signal TEMP_104_res_5 : std_logic;
	signal TEMP_104_res_6 : std_logic;
	signal TEMP_104_res_7 : std_logic;
	signal TEMP_105_res_0 : std_logic;
	signal TEMP_105_res_1 : std_logic;
	signal TEMP_105_res_2 : std_logic;
	signal TEMP_105_res_3 : std_logic;
	signal TEMP_106_res_0 : std_logic;
	signal TEMP_106_res_1 : std_logic;
	signal TEMP_106_in_2_0 : std_logic;
	signal TEMP_106_in_2_1 : std_logic;
	signal TEMP_106_in_2_2 : std_logic;
	signal TEMP_106_in_2_3 : std_logic;
	signal TEMP_106_in_2_4 : std_logic;
	signal TEMP_106_in_2_5 : std_logic;
	signal TEMP_106_in_2_6 : std_logic;
	signal TEMP_106_in_2_7 : std_logic;
	signal TEMP_106_in_2_8 : std_logic;
	signal TEMP_106_in_2_9 : std_logic;
	signal TEMP_106_out_2 : std_logic;
	signal TEMP_107_res_0 : std_logic;
	signal TEMP_107_res_1 : std_logic;
	signal TEMP_107_res_2 : std_logic;
	signal TEMP_107_res_3 : std_logic;
	signal TEMP_107_res_4 : std_logic;
	signal TEMP_107_res_5 : std_logic;
	signal TEMP_107_res_6 : std_logic;
	signal TEMP_107_res_7 : std_logic;
	signal TEMP_108_res_0 : std_logic;
	signal TEMP_108_res_1 : std_logic;
	signal TEMP_108_res_2 : std_logic;
	signal TEMP_108_res_3 : std_logic;
	signal TEMP_109_res_0 : std_logic;
	signal TEMP_109_res_1 : std_logic;
	signal TEMP_109_in_3_0 : std_logic;
	signal TEMP_109_in_3_1 : std_logic;
	signal TEMP_109_in_3_2 : std_logic;
	signal TEMP_109_in_3_3 : std_logic;
	signal TEMP_109_in_3_4 : std_logic;
	signal TEMP_109_in_3_5 : std_logic;
	signal TEMP_109_in_3_6 : std_logic;
	signal TEMP_109_in_3_7 : std_logic;
	signal TEMP_109_in_3_8 : std_logic;
	signal TEMP_109_in_3_9 : std_logic;
	signal TEMP_109_out_3 : std_logic;
	signal TEMP_110_res_0 : std_logic;
	signal TEMP_110_res_1 : std_logic;
	signal TEMP_110_res_2 : std_logic;
	signal TEMP_110_res_3 : std_logic;
	signal TEMP_110_res_4 : std_logic;
	signal TEMP_110_res_5 : std_logic;
	signal TEMP_110_res_6 : std_logic;
	signal TEMP_110_res_7 : std_logic;
	signal TEMP_111_res_0 : std_logic;
	signal TEMP_111_res_1 : std_logic;
	signal TEMP_111_res_2 : std_logic;
	signal TEMP_111_res_3 : std_logic;
	signal TEMP_112_res_0 : std_logic;
	signal TEMP_112_res_1 : std_logic;
	signal TEMP_113_mux_0 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_1 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_2 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_3 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_4 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_5 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_6 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_7 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_8 : std_logic_vector(9 downto 0);
	signal TEMP_113_mux_9 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_4 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_5 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_6 : std_logic_vector(9 downto 0);
	signal TEMP_114_res_7 : std_logic_vector(9 downto 0);
	signal TEMP_115_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_115_res_1 : std_logic_vector(9 downto 0);
	signal TEMP_115_res_2 : std_logic_vector(9 downto 0);
	signal TEMP_115_res_3 : std_logic_vector(9 downto 0);
	signal TEMP_116_res_0 : std_logic_vector(9 downto 0);
	signal TEMP_116_res_1 : std_logic_vector(9 downto 0);
	signal ldq_issue_set_vec_0 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_1 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_2 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_3 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_4 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_5 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_6 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_7 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_8 : std_logic_vector(0 downto 0);
	signal ldq_issue_set_vec_9 : std_logic_vector(0 downto 0);
	signal read_idx_oh_0_0 : std_logic;
	signal read_valid_0 : std_logic;
	signal read_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_117_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_0 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_118_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_119_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_120_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_120_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_120_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_120_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_121_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_121_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_1_0 : std_logic;
	signal read_valid_1 : std_logic;
	signal read_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_122_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_1 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_123_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_124_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_125_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_125_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_125_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_125_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_126_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_126_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_2_0 : std_logic;
	signal read_valid_2 : std_logic;
	signal read_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_127_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_2 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_128_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_129_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_130_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_130_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_130_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_130_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_131_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_131_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_3_0 : std_logic;
	signal read_valid_3 : std_logic;
	signal read_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_132_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_3 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_133_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_134_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_135_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_135_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_135_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_135_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_136_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_136_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_4_0 : std_logic;
	signal read_valid_4 : std_logic;
	signal read_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_137_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_4 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_138_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_139_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_140_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_140_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_140_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_140_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_141_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_141_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_5_0 : std_logic;
	signal read_valid_5 : std_logic;
	signal read_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_142_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_5 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_143_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_144_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_145_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_145_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_145_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_145_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_146_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_146_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_6_0 : std_logic;
	signal read_valid_6 : std_logic;
	signal read_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_147_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_6 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_148_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_149_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_150_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_150_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_150_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_150_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_151_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_151_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_7_0 : std_logic;
	signal read_valid_7 : std_logic;
	signal read_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_152_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_7 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_153_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_154_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_155_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_155_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_155_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_155_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_156_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_156_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_8_0 : std_logic;
	signal read_valid_8 : std_logic;
	signal read_data_8 : std_logic_vector(31 downto 0);
	signal TEMP_157_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_8 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_158_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_159_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_160_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_160_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_160_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_160_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_161_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_161_res_1 : std_logic_vector(31 downto 0);
	signal read_idx_oh_9_0 : std_logic;
	signal read_valid_9 : std_logic;
	signal read_data_9 : std_logic_vector(31 downto 0);
	signal TEMP_162_mux_0 : std_logic_vector(31 downto 0);
	signal bypass_data_9 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_0 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_1 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_2 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_3 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_4 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_5 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_6 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_7 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_8 : std_logic_vector(31 downto 0);
	signal TEMP_163_mux_9 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_4 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_5 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_6 : std_logic_vector(31 downto 0);
	signal TEMP_164_res_7 : std_logic_vector(31 downto 0);
	signal TEMP_165_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_165_res_1 : std_logic_vector(31 downto 0);
	signal TEMP_165_res_2 : std_logic_vector(31 downto 0);
	signal TEMP_165_res_3 : std_logic_vector(31 downto 0);
	signal TEMP_166_res_0 : std_logic_vector(31 downto 0);
	signal TEMP_166_res_1 : std_logic_vector(31 downto 0);
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
	ldq_head_oh(0) <= '1' when ldq_head_q = "0000" else '0';
	ldq_head_oh(1) <= '1' when ldq_head_q = "0001" else '0';
	ldq_head_oh(2) <= '1' when ldq_head_q = "0010" else '0';
	ldq_head_oh(3) <= '1' when ldq_head_q = "0011" else '0';
	ldq_head_oh(4) <= '1' when ldq_head_q = "0100" else '0';
	ldq_head_oh(5) <= '1' when ldq_head_q = "0101" else '0';
	ldq_head_oh(6) <= '1' when ldq_head_q = "0110" else '0';
	ldq_head_oh(7) <= '1' when ldq_head_q = "0111" else '0';
	ldq_head_oh(8) <= '1' when ldq_head_q = "1000" else '0';
	ldq_head_oh(9) <= '1' when ldq_head_q = "1001" else '0';
	-- Bits To One-Hot End

	-- Bits To One-Hot Begin
	-- BitsToOH(stq_head_oh, stq_head)
	stq_head_oh(0) <= '1' when stq_head_q = "0000" else '0';
	stq_head_oh(1) <= '1' when stq_head_q = "0001" else '0';
	stq_head_oh(2) <= '1' when stq_head_q = "0010" else '0';
	stq_head_oh(3) <= '1' when stq_head_q = "0011" else '0';
	stq_head_oh(4) <= '1' when stq_head_q = "0100" else '0';
	stq_head_oh(5) <= '1' when stq_head_q = "0101" else '0';
	stq_head_oh(6) <= '1' when stq_head_q = "0110" else '0';
	stq_head_oh(7) <= '1' when stq_head_q = "0111" else '0';
	stq_head_oh(8) <= '1' when stq_head_q = "1000" else '0';
	stq_head_oh(9) <= '1' when stq_head_q = "1001" else '0';
	-- Bits To One-Hot End

	ldq_alloc_next_0 <= not ldq_reset_0 and ldq_alloc_0_q;
	ldq_alloc_0_d <= ldq_wen_0 or ldq_alloc_next_0;
	ldq_issue_0_d <= not ldq_wen_0 and ( ldq_issue_set_0 or ldq_issue_0_q );
	ldq_addr_valid_0_d <= not ldq_wen_0 and ( ldq_addr_wen_0 or ldq_addr_valid_0_q );
	ldq_data_valid_0_d <= not ldq_wen_0 and ( ldq_data_wen_0 or ldq_data_valid_0_q );
	ldq_alloc_next_1 <= not ldq_reset_1 and ldq_alloc_1_q;
	ldq_alloc_1_d <= ldq_wen_1 or ldq_alloc_next_1;
	ldq_issue_1_d <= not ldq_wen_1 and ( ldq_issue_set_1 or ldq_issue_1_q );
	ldq_addr_valid_1_d <= not ldq_wen_1 and ( ldq_addr_wen_1 or ldq_addr_valid_1_q );
	ldq_data_valid_1_d <= not ldq_wen_1 and ( ldq_data_wen_1 or ldq_data_valid_1_q );
	ldq_alloc_next_2 <= not ldq_reset_2 and ldq_alloc_2_q;
	ldq_alloc_2_d <= ldq_wen_2 or ldq_alloc_next_2;
	ldq_issue_2_d <= not ldq_wen_2 and ( ldq_issue_set_2 or ldq_issue_2_q );
	ldq_addr_valid_2_d <= not ldq_wen_2 and ( ldq_addr_wen_2 or ldq_addr_valid_2_q );
	ldq_data_valid_2_d <= not ldq_wen_2 and ( ldq_data_wen_2 or ldq_data_valid_2_q );
	ldq_alloc_next_3 <= not ldq_reset_3 and ldq_alloc_3_q;
	ldq_alloc_3_d <= ldq_wen_3 or ldq_alloc_next_3;
	ldq_issue_3_d <= not ldq_wen_3 and ( ldq_issue_set_3 or ldq_issue_3_q );
	ldq_addr_valid_3_d <= not ldq_wen_3 and ( ldq_addr_wen_3 or ldq_addr_valid_3_q );
	ldq_data_valid_3_d <= not ldq_wen_3 and ( ldq_data_wen_3 or ldq_data_valid_3_q );
	ldq_alloc_next_4 <= not ldq_reset_4 and ldq_alloc_4_q;
	ldq_alloc_4_d <= ldq_wen_4 or ldq_alloc_next_4;
	ldq_issue_4_d <= not ldq_wen_4 and ( ldq_issue_set_4 or ldq_issue_4_q );
	ldq_addr_valid_4_d <= not ldq_wen_4 and ( ldq_addr_wen_4 or ldq_addr_valid_4_q );
	ldq_data_valid_4_d <= not ldq_wen_4 and ( ldq_data_wen_4 or ldq_data_valid_4_q );
	ldq_alloc_next_5 <= not ldq_reset_5 and ldq_alloc_5_q;
	ldq_alloc_5_d <= ldq_wen_5 or ldq_alloc_next_5;
	ldq_issue_5_d <= not ldq_wen_5 and ( ldq_issue_set_5 or ldq_issue_5_q );
	ldq_addr_valid_5_d <= not ldq_wen_5 and ( ldq_addr_wen_5 or ldq_addr_valid_5_q );
	ldq_data_valid_5_d <= not ldq_wen_5 and ( ldq_data_wen_5 or ldq_data_valid_5_q );
	ldq_alloc_next_6 <= not ldq_reset_6 and ldq_alloc_6_q;
	ldq_alloc_6_d <= ldq_wen_6 or ldq_alloc_next_6;
	ldq_issue_6_d <= not ldq_wen_6 and ( ldq_issue_set_6 or ldq_issue_6_q );
	ldq_addr_valid_6_d <= not ldq_wen_6 and ( ldq_addr_wen_6 or ldq_addr_valid_6_q );
	ldq_data_valid_6_d <= not ldq_wen_6 and ( ldq_data_wen_6 or ldq_data_valid_6_q );
	ldq_alloc_next_7 <= not ldq_reset_7 and ldq_alloc_7_q;
	ldq_alloc_7_d <= ldq_wen_7 or ldq_alloc_next_7;
	ldq_issue_7_d <= not ldq_wen_7 and ( ldq_issue_set_7 or ldq_issue_7_q );
	ldq_addr_valid_7_d <= not ldq_wen_7 and ( ldq_addr_wen_7 or ldq_addr_valid_7_q );
	ldq_data_valid_7_d <= not ldq_wen_7 and ( ldq_data_wen_7 or ldq_data_valid_7_q );
	ldq_alloc_next_8 <= not ldq_reset_8 and ldq_alloc_8_q;
	ldq_alloc_8_d <= ldq_wen_8 or ldq_alloc_next_8;
	ldq_issue_8_d <= not ldq_wen_8 and ( ldq_issue_set_8 or ldq_issue_8_q );
	ldq_addr_valid_8_d <= not ldq_wen_8 and ( ldq_addr_wen_8 or ldq_addr_valid_8_q );
	ldq_data_valid_8_d <= not ldq_wen_8 and ( ldq_data_wen_8 or ldq_data_valid_8_q );
	ldq_alloc_next_9 <= not ldq_reset_9 and ldq_alloc_9_q;
	ldq_alloc_9_d <= ldq_wen_9 or ldq_alloc_next_9;
	ldq_issue_9_d <= not ldq_wen_9 and ( ldq_issue_set_9 or ldq_issue_9_q );
	ldq_addr_valid_9_d <= not ldq_wen_9 and ( ldq_addr_wen_9 or ldq_addr_valid_9_q );
	ldq_data_valid_9_d <= not ldq_wen_9 and ( ldq_data_wen_9 or ldq_data_valid_9_q );
	stq_alloc_next_0 <= not stq_reset_0 and stq_alloc_0_q;
	stq_alloc_0_d <= stq_wen_0 or stq_alloc_next_0;
	stq_addr_valid_0_d <= not stq_wen_0 and ( stq_addr_wen_0 or stq_addr_valid_0_q );
	stq_data_valid_0_d <= not stq_wen_0 and ( stq_data_wen_0 or stq_data_valid_0_q );
	stq_alloc_next_1 <= not stq_reset_1 and stq_alloc_1_q;
	stq_alloc_1_d <= stq_wen_1 or stq_alloc_next_1;
	stq_addr_valid_1_d <= not stq_wen_1 and ( stq_addr_wen_1 or stq_addr_valid_1_q );
	stq_data_valid_1_d <= not stq_wen_1 and ( stq_data_wen_1 or stq_data_valid_1_q );
	stq_alloc_next_2 <= not stq_reset_2 and stq_alloc_2_q;
	stq_alloc_2_d <= stq_wen_2 or stq_alloc_next_2;
	stq_addr_valid_2_d <= not stq_wen_2 and ( stq_addr_wen_2 or stq_addr_valid_2_q );
	stq_data_valid_2_d <= not stq_wen_2 and ( stq_data_wen_2 or stq_data_valid_2_q );
	stq_alloc_next_3 <= not stq_reset_3 and stq_alloc_3_q;
	stq_alloc_3_d <= stq_wen_3 or stq_alloc_next_3;
	stq_addr_valid_3_d <= not stq_wen_3 and ( stq_addr_wen_3 or stq_addr_valid_3_q );
	stq_data_valid_3_d <= not stq_wen_3 and ( stq_data_wen_3 or stq_data_valid_3_q );
	stq_alloc_next_4 <= not stq_reset_4 and stq_alloc_4_q;
	stq_alloc_4_d <= stq_wen_4 or stq_alloc_next_4;
	stq_addr_valid_4_d <= not stq_wen_4 and ( stq_addr_wen_4 or stq_addr_valid_4_q );
	stq_data_valid_4_d <= not stq_wen_4 and ( stq_data_wen_4 or stq_data_valid_4_q );
	stq_alloc_next_5 <= not stq_reset_5 and stq_alloc_5_q;
	stq_alloc_5_d <= stq_wen_5 or stq_alloc_next_5;
	stq_addr_valid_5_d <= not stq_wen_5 and ( stq_addr_wen_5 or stq_addr_valid_5_q );
	stq_data_valid_5_d <= not stq_wen_5 and ( stq_data_wen_5 or stq_data_valid_5_q );
	stq_alloc_next_6 <= not stq_reset_6 and stq_alloc_6_q;
	stq_alloc_6_d <= stq_wen_6 or stq_alloc_next_6;
	stq_addr_valid_6_d <= not stq_wen_6 and ( stq_addr_wen_6 or stq_addr_valid_6_q );
	stq_data_valid_6_d <= not stq_wen_6 and ( stq_data_wen_6 or stq_data_valid_6_q );
	stq_alloc_next_7 <= not stq_reset_7 and stq_alloc_7_q;
	stq_alloc_7_d <= stq_wen_7 or stq_alloc_next_7;
	stq_addr_valid_7_d <= not stq_wen_7 and ( stq_addr_wen_7 or stq_addr_valid_7_q );
	stq_data_valid_7_d <= not stq_wen_7 and ( stq_data_wen_7 or stq_data_valid_7_q );
	stq_alloc_next_8 <= not stq_reset_8 and stq_alloc_8_q;
	stq_alloc_8_d <= stq_wen_8 or stq_alloc_next_8;
	stq_addr_valid_8_d <= not stq_wen_8 and ( stq_addr_wen_8 or stq_addr_valid_8_q );
	stq_data_valid_8_d <= not stq_wen_8 and ( stq_data_wen_8 or stq_data_valid_8_q );
	stq_alloc_next_9 <= not stq_reset_9 and stq_alloc_9_q;
	stq_alloc_9_d <= stq_wen_9 or stq_alloc_next_9;
	stq_addr_valid_9_d <= not stq_wen_9 and ( stq_addr_wen_9 or stq_addr_valid_9_q );
	stq_data_valid_9_d <= not stq_wen_9 and ( stq_data_wen_9 or stq_data_valid_9_q );
	store_is_older_0_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_0(0) ) ) when ldq_wen_0 else not stq_reset_0 and store_is_older_0_q(0);
	store_is_older_0_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_0(1) ) ) when ldq_wen_0 else not stq_reset_1 and store_is_older_0_q(1);
	store_is_older_0_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_0(2) ) ) when ldq_wen_0 else not stq_reset_2 and store_is_older_0_q(2);
	store_is_older_0_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_0(3) ) ) when ldq_wen_0 else not stq_reset_3 and store_is_older_0_q(3);
	store_is_older_0_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_0(4) ) ) when ldq_wen_0 else not stq_reset_4 and store_is_older_0_q(4);
	store_is_older_0_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_0(5) ) ) when ldq_wen_0 else not stq_reset_5 and store_is_older_0_q(5);
	store_is_older_0_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_0(6) ) ) when ldq_wen_0 else not stq_reset_6 and store_is_older_0_q(6);
	store_is_older_0_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_0(7) ) ) when ldq_wen_0 else not stq_reset_7 and store_is_older_0_q(7);
	store_is_older_0_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_0(8) ) ) when ldq_wen_0 else not stq_reset_8 and store_is_older_0_q(8);
	store_is_older_0_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_0(9) ) ) when ldq_wen_0 else not stq_reset_9 and store_is_older_0_q(9);
	store_is_older_1_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_1(0) ) ) when ldq_wen_1 else not stq_reset_0 and store_is_older_1_q(0);
	store_is_older_1_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_1(1) ) ) when ldq_wen_1 else not stq_reset_1 and store_is_older_1_q(1);
	store_is_older_1_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_1(2) ) ) when ldq_wen_1 else not stq_reset_2 and store_is_older_1_q(2);
	store_is_older_1_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_1(3) ) ) when ldq_wen_1 else not stq_reset_3 and store_is_older_1_q(3);
	store_is_older_1_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_1(4) ) ) when ldq_wen_1 else not stq_reset_4 and store_is_older_1_q(4);
	store_is_older_1_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_1(5) ) ) when ldq_wen_1 else not stq_reset_5 and store_is_older_1_q(5);
	store_is_older_1_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_1(6) ) ) when ldq_wen_1 else not stq_reset_6 and store_is_older_1_q(6);
	store_is_older_1_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_1(7) ) ) when ldq_wen_1 else not stq_reset_7 and store_is_older_1_q(7);
	store_is_older_1_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_1(8) ) ) when ldq_wen_1 else not stq_reset_8 and store_is_older_1_q(8);
	store_is_older_1_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_1(9) ) ) when ldq_wen_1 else not stq_reset_9 and store_is_older_1_q(9);
	store_is_older_2_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_2(0) ) ) when ldq_wen_2 else not stq_reset_0 and store_is_older_2_q(0);
	store_is_older_2_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_2(1) ) ) when ldq_wen_2 else not stq_reset_1 and store_is_older_2_q(1);
	store_is_older_2_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_2(2) ) ) when ldq_wen_2 else not stq_reset_2 and store_is_older_2_q(2);
	store_is_older_2_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_2(3) ) ) when ldq_wen_2 else not stq_reset_3 and store_is_older_2_q(3);
	store_is_older_2_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_2(4) ) ) when ldq_wen_2 else not stq_reset_4 and store_is_older_2_q(4);
	store_is_older_2_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_2(5) ) ) when ldq_wen_2 else not stq_reset_5 and store_is_older_2_q(5);
	store_is_older_2_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_2(6) ) ) when ldq_wen_2 else not stq_reset_6 and store_is_older_2_q(6);
	store_is_older_2_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_2(7) ) ) when ldq_wen_2 else not stq_reset_7 and store_is_older_2_q(7);
	store_is_older_2_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_2(8) ) ) when ldq_wen_2 else not stq_reset_8 and store_is_older_2_q(8);
	store_is_older_2_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_2(9) ) ) when ldq_wen_2 else not stq_reset_9 and store_is_older_2_q(9);
	store_is_older_3_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_3(0) ) ) when ldq_wen_3 else not stq_reset_0 and store_is_older_3_q(0);
	store_is_older_3_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_3(1) ) ) when ldq_wen_3 else not stq_reset_1 and store_is_older_3_q(1);
	store_is_older_3_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_3(2) ) ) when ldq_wen_3 else not stq_reset_2 and store_is_older_3_q(2);
	store_is_older_3_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_3(3) ) ) when ldq_wen_3 else not stq_reset_3 and store_is_older_3_q(3);
	store_is_older_3_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_3(4) ) ) when ldq_wen_3 else not stq_reset_4 and store_is_older_3_q(4);
	store_is_older_3_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_3(5) ) ) when ldq_wen_3 else not stq_reset_5 and store_is_older_3_q(5);
	store_is_older_3_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_3(6) ) ) when ldq_wen_3 else not stq_reset_6 and store_is_older_3_q(6);
	store_is_older_3_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_3(7) ) ) when ldq_wen_3 else not stq_reset_7 and store_is_older_3_q(7);
	store_is_older_3_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_3(8) ) ) when ldq_wen_3 else not stq_reset_8 and store_is_older_3_q(8);
	store_is_older_3_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_3(9) ) ) when ldq_wen_3 else not stq_reset_9 and store_is_older_3_q(9);
	store_is_older_4_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_4(0) ) ) when ldq_wen_4 else not stq_reset_0 and store_is_older_4_q(0);
	store_is_older_4_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_4(1) ) ) when ldq_wen_4 else not stq_reset_1 and store_is_older_4_q(1);
	store_is_older_4_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_4(2) ) ) when ldq_wen_4 else not stq_reset_2 and store_is_older_4_q(2);
	store_is_older_4_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_4(3) ) ) when ldq_wen_4 else not stq_reset_3 and store_is_older_4_q(3);
	store_is_older_4_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_4(4) ) ) when ldq_wen_4 else not stq_reset_4 and store_is_older_4_q(4);
	store_is_older_4_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_4(5) ) ) when ldq_wen_4 else not stq_reset_5 and store_is_older_4_q(5);
	store_is_older_4_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_4(6) ) ) when ldq_wen_4 else not stq_reset_6 and store_is_older_4_q(6);
	store_is_older_4_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_4(7) ) ) when ldq_wen_4 else not stq_reset_7 and store_is_older_4_q(7);
	store_is_older_4_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_4(8) ) ) when ldq_wen_4 else not stq_reset_8 and store_is_older_4_q(8);
	store_is_older_4_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_4(9) ) ) when ldq_wen_4 else not stq_reset_9 and store_is_older_4_q(9);
	store_is_older_5_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_5(0) ) ) when ldq_wen_5 else not stq_reset_0 and store_is_older_5_q(0);
	store_is_older_5_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_5(1) ) ) when ldq_wen_5 else not stq_reset_1 and store_is_older_5_q(1);
	store_is_older_5_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_5(2) ) ) when ldq_wen_5 else not stq_reset_2 and store_is_older_5_q(2);
	store_is_older_5_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_5(3) ) ) when ldq_wen_5 else not stq_reset_3 and store_is_older_5_q(3);
	store_is_older_5_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_5(4) ) ) when ldq_wen_5 else not stq_reset_4 and store_is_older_5_q(4);
	store_is_older_5_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_5(5) ) ) when ldq_wen_5 else not stq_reset_5 and store_is_older_5_q(5);
	store_is_older_5_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_5(6) ) ) when ldq_wen_5 else not stq_reset_6 and store_is_older_5_q(6);
	store_is_older_5_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_5(7) ) ) when ldq_wen_5 else not stq_reset_7 and store_is_older_5_q(7);
	store_is_older_5_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_5(8) ) ) when ldq_wen_5 else not stq_reset_8 and store_is_older_5_q(8);
	store_is_older_5_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_5(9) ) ) when ldq_wen_5 else not stq_reset_9 and store_is_older_5_q(9);
	store_is_older_6_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_6(0) ) ) when ldq_wen_6 else not stq_reset_0 and store_is_older_6_q(0);
	store_is_older_6_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_6(1) ) ) when ldq_wen_6 else not stq_reset_1 and store_is_older_6_q(1);
	store_is_older_6_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_6(2) ) ) when ldq_wen_6 else not stq_reset_2 and store_is_older_6_q(2);
	store_is_older_6_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_6(3) ) ) when ldq_wen_6 else not stq_reset_3 and store_is_older_6_q(3);
	store_is_older_6_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_6(4) ) ) when ldq_wen_6 else not stq_reset_4 and store_is_older_6_q(4);
	store_is_older_6_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_6(5) ) ) when ldq_wen_6 else not stq_reset_5 and store_is_older_6_q(5);
	store_is_older_6_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_6(6) ) ) when ldq_wen_6 else not stq_reset_6 and store_is_older_6_q(6);
	store_is_older_6_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_6(7) ) ) when ldq_wen_6 else not stq_reset_7 and store_is_older_6_q(7);
	store_is_older_6_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_6(8) ) ) when ldq_wen_6 else not stq_reset_8 and store_is_older_6_q(8);
	store_is_older_6_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_6(9) ) ) when ldq_wen_6 else not stq_reset_9 and store_is_older_6_q(9);
	store_is_older_7_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_7(0) ) ) when ldq_wen_7 else not stq_reset_0 and store_is_older_7_q(0);
	store_is_older_7_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_7(1) ) ) when ldq_wen_7 else not stq_reset_1 and store_is_older_7_q(1);
	store_is_older_7_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_7(2) ) ) when ldq_wen_7 else not stq_reset_2 and store_is_older_7_q(2);
	store_is_older_7_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_7(3) ) ) when ldq_wen_7 else not stq_reset_3 and store_is_older_7_q(3);
	store_is_older_7_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_7(4) ) ) when ldq_wen_7 else not stq_reset_4 and store_is_older_7_q(4);
	store_is_older_7_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_7(5) ) ) when ldq_wen_7 else not stq_reset_5 and store_is_older_7_q(5);
	store_is_older_7_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_7(6) ) ) when ldq_wen_7 else not stq_reset_6 and store_is_older_7_q(6);
	store_is_older_7_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_7(7) ) ) when ldq_wen_7 else not stq_reset_7 and store_is_older_7_q(7);
	store_is_older_7_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_7(8) ) ) when ldq_wen_7 else not stq_reset_8 and store_is_older_7_q(8);
	store_is_older_7_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_7(9) ) ) when ldq_wen_7 else not stq_reset_9 and store_is_older_7_q(9);
	store_is_older_8_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_8(0) ) ) when ldq_wen_8 else not stq_reset_0 and store_is_older_8_q(0);
	store_is_older_8_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_8(1) ) ) when ldq_wen_8 else not stq_reset_1 and store_is_older_8_q(1);
	store_is_older_8_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_8(2) ) ) when ldq_wen_8 else not stq_reset_2 and store_is_older_8_q(2);
	store_is_older_8_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_8(3) ) ) when ldq_wen_8 else not stq_reset_3 and store_is_older_8_q(3);
	store_is_older_8_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_8(4) ) ) when ldq_wen_8 else not stq_reset_4 and store_is_older_8_q(4);
	store_is_older_8_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_8(5) ) ) when ldq_wen_8 else not stq_reset_5 and store_is_older_8_q(5);
	store_is_older_8_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_8(6) ) ) when ldq_wen_8 else not stq_reset_6 and store_is_older_8_q(6);
	store_is_older_8_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_8(7) ) ) when ldq_wen_8 else not stq_reset_7 and store_is_older_8_q(7);
	store_is_older_8_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_8(8) ) ) when ldq_wen_8 else not stq_reset_8 and store_is_older_8_q(8);
	store_is_older_8_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_8(9) ) ) when ldq_wen_8 else not stq_reset_9 and store_is_older_8_q(9);
	store_is_older_9_d(0) <= ( not stq_reset_0 and ( stq_alloc_0_q or ga_ls_order_9(0) ) ) when ldq_wen_9 else not stq_reset_0 and store_is_older_9_q(0);
	store_is_older_9_d(1) <= ( not stq_reset_1 and ( stq_alloc_1_q or ga_ls_order_9(1) ) ) when ldq_wen_9 else not stq_reset_1 and store_is_older_9_q(1);
	store_is_older_9_d(2) <= ( not stq_reset_2 and ( stq_alloc_2_q or ga_ls_order_9(2) ) ) when ldq_wen_9 else not stq_reset_2 and store_is_older_9_q(2);
	store_is_older_9_d(3) <= ( not stq_reset_3 and ( stq_alloc_3_q or ga_ls_order_9(3) ) ) when ldq_wen_9 else not stq_reset_3 and store_is_older_9_q(3);
	store_is_older_9_d(4) <= ( not stq_reset_4 and ( stq_alloc_4_q or ga_ls_order_9(4) ) ) when ldq_wen_9 else not stq_reset_4 and store_is_older_9_q(4);
	store_is_older_9_d(5) <= ( not stq_reset_5 and ( stq_alloc_5_q or ga_ls_order_9(5) ) ) when ldq_wen_9 else not stq_reset_5 and store_is_older_9_q(5);
	store_is_older_9_d(6) <= ( not stq_reset_6 and ( stq_alloc_6_q or ga_ls_order_9(6) ) ) when ldq_wen_9 else not stq_reset_6 and store_is_older_9_q(6);
	store_is_older_9_d(7) <= ( not stq_reset_7 and ( stq_alloc_7_q or ga_ls_order_9(7) ) ) when ldq_wen_9 else not stq_reset_7 and store_is_older_9_q(7);
	store_is_older_9_d(8) <= ( not stq_reset_8 and ( stq_alloc_8_q or ga_ls_order_9(8) ) ) when ldq_wen_9 else not stq_reset_8 and store_is_older_9_q(8);
	store_is_older_9_d(9) <= ( not stq_reset_9 and ( stq_alloc_9_q or ga_ls_order_9(9) ) ) when ldq_wen_9 else not stq_reset_9 and store_is_older_9_q(9);
	-- Reduction Begin
	-- Reduce(ldq_not_empty, ldq_alloc, or)
	TEMP_1_res_0 <= ldq_alloc_0_q or ldq_alloc_8_q;
	TEMP_1_res_1 <= ldq_alloc_1_q or ldq_alloc_9_q;
	TEMP_1_res_2 <= ldq_alloc_2_q;
	TEMP_1_res_3 <= ldq_alloc_3_q;
	TEMP_1_res_4 <= ldq_alloc_4_q;
	TEMP_1_res_5 <= ldq_alloc_5_q;
	TEMP_1_res_6 <= ldq_alloc_6_q;
	TEMP_1_res_7 <= ldq_alloc_7_q;
	-- Layer End
	TEMP_2_res_0 <= TEMP_1_res_0 or TEMP_1_res_4;
	TEMP_2_res_1 <= TEMP_1_res_1 or TEMP_1_res_5;
	TEMP_2_res_2 <= TEMP_1_res_2 or TEMP_1_res_6;
	TEMP_2_res_3 <= TEMP_1_res_3 or TEMP_1_res_7;
	-- Layer End
	TEMP_3_res_0 <= TEMP_2_res_0 or TEMP_2_res_2;
	TEMP_3_res_1 <= TEMP_2_res_1 or TEMP_2_res_3;
	-- Layer End
	ldq_not_empty <= TEMP_3_res_0 or TEMP_3_res_1;
	-- Reduction End

	ldq_empty <= not ldq_not_empty;
	-- MuxLookUp Begin
	-- MuxLookUp(stq_not_empty, stq_alloc, stq_head)
	stq_not_empty <= 
	stq_alloc_0_q when (stq_head_q = "0000") else
	stq_alloc_1_q when (stq_head_q = "0001") else
	stq_alloc_2_q when (stq_head_q = "0010") else
	stq_alloc_3_q when (stq_head_q = "0011") else
	stq_alloc_4_q when (stq_head_q = "0100") else
	stq_alloc_5_q when (stq_head_q = "0101") else
	stq_alloc_6_q when (stq_head_q = "0110") else
	stq_alloc_7_q when (stq_head_q = "0111") else
	stq_alloc_8_q when (stq_head_q = "1000") else
	stq_alloc_9_q when (stq_head_q = "1001") else
	'0';
	-- MuxLookUp End

	stq_empty <= not stq_not_empty;
	empty_o <= ldq_empty and stq_empty;
	-- WrapAdd Begin
	-- WrapAdd(ldq_tail, ldq_tail, num_loads, 10)
	TEMP_4_sum <= std_logic_vector(unsigned('0' & ldq_tail_q) + unsigned('0' & num_loads));
	TEMP_4_res <= std_logic_vector(unsigned(TEMP_4_sum) - 10) when unsigned(TEMP_4_sum) >= 10 else TEMP_4_sum;
	ldq_tail_d <= TEMP_4_res(3 downto 0);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_tail, stq_tail, num_stores, 10)
	TEMP_5_sum <= std_logic_vector(unsigned('0' & stq_tail_q) + unsigned('0' & num_stores));
	TEMP_5_res <= std_logic_vector(unsigned(TEMP_5_sum) - 10) when unsigned(TEMP_5_sum) >= 10 else TEMP_5_sum;
	stq_tail_d <= TEMP_5_res(3 downto 0);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_issue, stq_issue, 1, 10)
	stq_issue_d <= std_logic_vector(unsigned(stq_issue_q) - 9) when unsigned(stq_issue_q) >= 9 else std_logic_vector(unsigned(stq_issue_q) + 1);
	-- WrapAdd End

	-- WrapAdd Begin
	-- WrapAdd(stq_resp, stq_resp, 1, 10)
	stq_resp_d <= std_logic_vector(unsigned(stq_resp_q) - 9) when unsigned(stq_resp_q) >= 9 else std_logic_vector(unsigned(stq_resp_q) + 1);
	-- WrapAdd End

	-- Bits To One-Hot Begin
	-- BitsToOH(ldq_tail_oh, ldq_tail)
	ldq_tail_oh(0) <= '1' when ldq_tail_q = "0000" else '0';
	ldq_tail_oh(1) <= '1' when ldq_tail_q = "0001" else '0';
	ldq_tail_oh(2) <= '1' when ldq_tail_q = "0010" else '0';
	ldq_tail_oh(3) <= '1' when ldq_tail_q = "0011" else '0';
	ldq_tail_oh(4) <= '1' when ldq_tail_q = "0100" else '0';
	ldq_tail_oh(5) <= '1' when ldq_tail_q = "0101" else '0';
	ldq_tail_oh(6) <= '1' when ldq_tail_q = "0110" else '0';
	ldq_tail_oh(7) <= '1' when ldq_tail_q = "0111" else '0';
	ldq_tail_oh(8) <= '1' when ldq_tail_q = "1000" else '0';
	ldq_tail_oh(9) <= '1' when ldq_tail_q = "1001" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
	TEMP_6_double_in(0) <= ldq_alloc_next_0;
	TEMP_6_double_in(10) <= ldq_alloc_next_0;
	TEMP_6_double_in(1) <= ldq_alloc_next_1;
	TEMP_6_double_in(11) <= ldq_alloc_next_1;
	TEMP_6_double_in(2) <= ldq_alloc_next_2;
	TEMP_6_double_in(12) <= ldq_alloc_next_2;
	TEMP_6_double_in(3) <= ldq_alloc_next_3;
	TEMP_6_double_in(13) <= ldq_alloc_next_3;
	TEMP_6_double_in(4) <= ldq_alloc_next_4;
	TEMP_6_double_in(14) <= ldq_alloc_next_4;
	TEMP_6_double_in(5) <= ldq_alloc_next_5;
	TEMP_6_double_in(15) <= ldq_alloc_next_5;
	TEMP_6_double_in(6) <= ldq_alloc_next_6;
	TEMP_6_double_in(16) <= ldq_alloc_next_6;
	TEMP_6_double_in(7) <= ldq_alloc_next_7;
	TEMP_6_double_in(17) <= ldq_alloc_next_7;
	TEMP_6_double_in(8) <= ldq_alloc_next_8;
	TEMP_6_double_in(18) <= ldq_alloc_next_8;
	TEMP_6_double_in(9) <= ldq_alloc_next_9;
	TEMP_6_double_in(19) <= ldq_alloc_next_9;
	TEMP_6_double_out <= TEMP_6_double_in and not std_logic_vector( unsigned( TEMP_6_double_in ) - unsigned( "0000000000" & ldq_tail_oh ) );
	ldq_head_next_oh <= TEMP_6_double_out(9 downto 0) or TEMP_6_double_out(19 downto 10);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(ldq_head_sel, ldq_alloc_next, or)
	TEMP_7_res_0 <= ldq_alloc_next_0 or ldq_alloc_next_8;
	TEMP_7_res_1 <= ldq_alloc_next_1 or ldq_alloc_next_9;
	TEMP_7_res_2 <= ldq_alloc_next_2;
	TEMP_7_res_3 <= ldq_alloc_next_3;
	TEMP_7_res_4 <= ldq_alloc_next_4;
	TEMP_7_res_5 <= ldq_alloc_next_5;
	TEMP_7_res_6 <= ldq_alloc_next_6;
	TEMP_7_res_7 <= ldq_alloc_next_7;
	-- Layer End
	TEMP_8_res_0 <= TEMP_7_res_0 or TEMP_7_res_4;
	TEMP_8_res_1 <= TEMP_7_res_1 or TEMP_7_res_5;
	TEMP_8_res_2 <= TEMP_7_res_2 or TEMP_7_res_6;
	TEMP_8_res_3 <= TEMP_7_res_3 or TEMP_7_res_7;
	-- Layer End
	TEMP_9_res_0 <= TEMP_8_res_0 or TEMP_8_res_2;
	TEMP_9_res_1 <= TEMP_8_res_1 or TEMP_8_res_3;
	-- Layer End
	ldq_head_sel <= TEMP_9_res_0 or TEMP_9_res_1;
	-- Reduction End

	-- One-Hot To Bits Begin
	-- OHToBits(ldq_head_next, ldq_head_next_oh)
	TEMP_10_in_0_0 <= '0';
	TEMP_10_in_0_1 <= ldq_head_next_oh(1);
	TEMP_10_in_0_2 <= '0';
	TEMP_10_in_0_3 <= ldq_head_next_oh(3);
	TEMP_10_in_0_4 <= '0';
	TEMP_10_in_0_5 <= ldq_head_next_oh(5);
	TEMP_10_in_0_6 <= '0';
	TEMP_10_in_0_7 <= ldq_head_next_oh(7);
	TEMP_10_in_0_8 <= '0';
	TEMP_10_in_0_9 <= ldq_head_next_oh(9);
	TEMP_11_res_0 <= TEMP_10_in_0_0 or TEMP_10_in_0_8;
	TEMP_11_res_1 <= TEMP_10_in_0_1 or TEMP_10_in_0_9;
	TEMP_11_res_2 <= TEMP_10_in_0_2;
	TEMP_11_res_3 <= TEMP_10_in_0_3;
	TEMP_11_res_4 <= TEMP_10_in_0_4;
	TEMP_11_res_5 <= TEMP_10_in_0_5;
	TEMP_11_res_6 <= TEMP_10_in_0_6;
	TEMP_11_res_7 <= TEMP_10_in_0_7;
	-- Layer End
	TEMP_12_res_0 <= TEMP_11_res_0 or TEMP_11_res_4;
	TEMP_12_res_1 <= TEMP_11_res_1 or TEMP_11_res_5;
	TEMP_12_res_2 <= TEMP_11_res_2 or TEMP_11_res_6;
	TEMP_12_res_3 <= TEMP_11_res_3 or TEMP_11_res_7;
	-- Layer End
	TEMP_13_res_0 <= TEMP_12_res_0 or TEMP_12_res_2;
	TEMP_13_res_1 <= TEMP_12_res_1 or TEMP_12_res_3;
	-- Layer End
	TEMP_10_out_0 <= TEMP_13_res_0 or TEMP_13_res_1;
	ldq_head_next(0) <= TEMP_10_out_0;
	TEMP_13_in_1_0 <= '0';
	TEMP_13_in_1_1 <= '0';
	TEMP_13_in_1_2 <= ldq_head_next_oh(2);
	TEMP_13_in_1_3 <= ldq_head_next_oh(3);
	TEMP_13_in_1_4 <= '0';
	TEMP_13_in_1_5 <= '0';
	TEMP_13_in_1_6 <= ldq_head_next_oh(6);
	TEMP_13_in_1_7 <= ldq_head_next_oh(7);
	TEMP_13_in_1_8 <= '0';
	TEMP_13_in_1_9 <= '0';
	TEMP_14_res_0 <= TEMP_13_in_1_0 or TEMP_13_in_1_8;
	TEMP_14_res_1 <= TEMP_13_in_1_1 or TEMP_13_in_1_9;
	TEMP_14_res_2 <= TEMP_13_in_1_2;
	TEMP_14_res_3 <= TEMP_13_in_1_3;
	TEMP_14_res_4 <= TEMP_13_in_1_4;
	TEMP_14_res_5 <= TEMP_13_in_1_5;
	TEMP_14_res_6 <= TEMP_13_in_1_6;
	TEMP_14_res_7 <= TEMP_13_in_1_7;
	-- Layer End
	TEMP_15_res_0 <= TEMP_14_res_0 or TEMP_14_res_4;
	TEMP_15_res_1 <= TEMP_14_res_1 or TEMP_14_res_5;
	TEMP_15_res_2 <= TEMP_14_res_2 or TEMP_14_res_6;
	TEMP_15_res_3 <= TEMP_14_res_3 or TEMP_14_res_7;
	-- Layer End
	TEMP_16_res_0 <= TEMP_15_res_0 or TEMP_15_res_2;
	TEMP_16_res_1 <= TEMP_15_res_1 or TEMP_15_res_3;
	-- Layer End
	TEMP_13_out_1 <= TEMP_16_res_0 or TEMP_16_res_1;
	ldq_head_next(1) <= TEMP_13_out_1;
	TEMP_16_in_2_0 <= '0';
	TEMP_16_in_2_1 <= '0';
	TEMP_16_in_2_2 <= '0';
	TEMP_16_in_2_3 <= '0';
	TEMP_16_in_2_4 <= ldq_head_next_oh(4);
	TEMP_16_in_2_5 <= ldq_head_next_oh(5);
	TEMP_16_in_2_6 <= ldq_head_next_oh(6);
	TEMP_16_in_2_7 <= ldq_head_next_oh(7);
	TEMP_16_in_2_8 <= '0';
	TEMP_16_in_2_9 <= '0';
	TEMP_17_res_0 <= TEMP_16_in_2_0 or TEMP_16_in_2_8;
	TEMP_17_res_1 <= TEMP_16_in_2_1 or TEMP_16_in_2_9;
	TEMP_17_res_2 <= TEMP_16_in_2_2;
	TEMP_17_res_3 <= TEMP_16_in_2_3;
	TEMP_17_res_4 <= TEMP_16_in_2_4;
	TEMP_17_res_5 <= TEMP_16_in_2_5;
	TEMP_17_res_6 <= TEMP_16_in_2_6;
	TEMP_17_res_7 <= TEMP_16_in_2_7;
	-- Layer End
	TEMP_18_res_0 <= TEMP_17_res_0 or TEMP_17_res_4;
	TEMP_18_res_1 <= TEMP_17_res_1 or TEMP_17_res_5;
	TEMP_18_res_2 <= TEMP_17_res_2 or TEMP_17_res_6;
	TEMP_18_res_3 <= TEMP_17_res_3 or TEMP_17_res_7;
	-- Layer End
	TEMP_19_res_0 <= TEMP_18_res_0 or TEMP_18_res_2;
	TEMP_19_res_1 <= TEMP_18_res_1 or TEMP_18_res_3;
	-- Layer End
	TEMP_16_out_2 <= TEMP_19_res_0 or TEMP_19_res_1;
	ldq_head_next(2) <= TEMP_16_out_2;
	TEMP_19_in_3_0 <= '0';
	TEMP_19_in_3_1 <= '0';
	TEMP_19_in_3_2 <= '0';
	TEMP_19_in_3_3 <= '0';
	TEMP_19_in_3_4 <= '0';
	TEMP_19_in_3_5 <= '0';
	TEMP_19_in_3_6 <= '0';
	TEMP_19_in_3_7 <= '0';
	TEMP_19_in_3_8 <= ldq_head_next_oh(8);
	TEMP_19_in_3_9 <= ldq_head_next_oh(9);
	TEMP_20_res_0 <= TEMP_19_in_3_0 or TEMP_19_in_3_8;
	TEMP_20_res_1 <= TEMP_19_in_3_1 or TEMP_19_in_3_9;
	TEMP_20_res_2 <= TEMP_19_in_3_2;
	TEMP_20_res_3 <= TEMP_19_in_3_3;
	TEMP_20_res_4 <= TEMP_19_in_3_4;
	TEMP_20_res_5 <= TEMP_19_in_3_5;
	TEMP_20_res_6 <= TEMP_19_in_3_6;
	TEMP_20_res_7 <= TEMP_19_in_3_7;
	-- Layer End
	TEMP_21_res_0 <= TEMP_20_res_0 or TEMP_20_res_4;
	TEMP_21_res_1 <= TEMP_20_res_1 or TEMP_20_res_5;
	TEMP_21_res_2 <= TEMP_20_res_2 or TEMP_20_res_6;
	TEMP_21_res_3 <= TEMP_20_res_3 or TEMP_20_res_7;
	-- Layer End
	TEMP_22_res_0 <= TEMP_21_res_0 or TEMP_21_res_2;
	TEMP_22_res_1 <= TEMP_21_res_1 or TEMP_21_res_3;
	-- Layer End
	TEMP_19_out_3 <= TEMP_22_res_0 or TEMP_22_res_1;
	ldq_head_next(3) <= TEMP_19_out_3;
	-- One-Hot To Bits End

	ldq_head_d <= ldq_head_next when ldq_head_sel else ldq_tail_q;
	-- Bits To One-Hot Begin
	-- BitsToOH(stq_tail_oh, stq_tail)
	stq_tail_oh(0) <= '1' when stq_tail_q = "0000" else '0';
	stq_tail_oh(1) <= '1' when stq_tail_q = "0001" else '0';
	stq_tail_oh(2) <= '1' when stq_tail_q = "0010" else '0';
	stq_tail_oh(3) <= '1' when stq_tail_q = "0011" else '0';
	stq_tail_oh(4) <= '1' when stq_tail_q = "0100" else '0';
	stq_tail_oh(5) <= '1' when stq_tail_q = "0101" else '0';
	stq_tail_oh(6) <= '1' when stq_tail_q = "0110" else '0';
	stq_tail_oh(7) <= '1' when stq_tail_q = "0111" else '0';
	stq_tail_oh(8) <= '1' when stq_tail_q = "1000" else '0';
	stq_tail_oh(9) <= '1' when stq_tail_q = "1001" else '0';
	-- Bits To One-Hot End

	-- WrapAdd Begin
	-- WrapAdd(stq_head_next, stq_head, 1, 10)
	stq_head_next <= std_logic_vector(unsigned(stq_head_q) - 9) when unsigned(stq_head_q) >= 9 else std_logic_vector(unsigned(stq_head_q) + 1);
	-- WrapAdd End

	stq_head_sel <= wresp_valid_0_i;
	stq_head_d <= stq_head_next when stq_head_sel else stq_head_q;
	handshake_lsq_lsq1_core_ga : entity work.handshake_lsq_lsq1_core_ga
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
			ldq_wen_2_o => ldq_wen_2,
			ldq_wen_3_o => ldq_wen_3,
			ldq_wen_4_o => ldq_wen_4,
			ldq_wen_5_o => ldq_wen_5,
			ldq_wen_6_o => ldq_wen_6,
			ldq_wen_7_o => ldq_wen_7,
			ldq_wen_8_o => ldq_wen_8,
			ldq_wen_9_o => ldq_wen_9,
			num_loads_o => num_loads,
			stq_wen_0_o => stq_wen_0,
			stq_wen_1_o => stq_wen_1,
			stq_wen_2_o => stq_wen_2,
			stq_wen_3_o => stq_wen_3,
			stq_wen_4_o => stq_wen_4,
			stq_wen_5_o => stq_wen_5,
			stq_wen_6_o => stq_wen_6,
			stq_wen_7_o => stq_wen_7,
			stq_wen_8_o => stq_wen_8,
			stq_wen_9_o => stq_wen_9,
			ga_ls_order_0_o => ga_ls_order_0,
			ga_ls_order_1_o => ga_ls_order_1,
			ga_ls_order_2_o => ga_ls_order_2,
			ga_ls_order_3_o => ga_ls_order_3,
			ga_ls_order_4_o => ga_ls_order_4,
			ga_ls_order_5_o => ga_ls_order_5,
			ga_ls_order_6_o => ga_ls_order_6,
			ga_ls_order_7_o => ga_ls_order_7,
			ga_ls_order_8_o => ga_ls_order_8,
			ga_ls_order_9_o => ga_ls_order_9,
			num_stores_o => num_stores
		);
	handshake_lsq_lsq1_core_lda_dispatcher : entity work.handshake_lsq_lsq1_core_lda
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => ldp_addr_0_i,
			port_ready_0_o => ldp_addr_ready_0_o,
			port_valid_0_i => ldp_addr_valid_0_i,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_alloc_8_i => ldq_alloc_8_q,
			entry_alloc_9_i => ldq_alloc_9_q,
			entry_payload_valid_0_i => ldq_addr_valid_0_q,
			entry_payload_valid_1_i => ldq_addr_valid_1_q,
			entry_payload_valid_2_i => ldq_addr_valid_2_q,
			entry_payload_valid_3_i => ldq_addr_valid_3_q,
			entry_payload_valid_4_i => ldq_addr_valid_4_q,
			entry_payload_valid_5_i => ldq_addr_valid_5_q,
			entry_payload_valid_6_i => ldq_addr_valid_6_q,
			entry_payload_valid_7_i => ldq_addr_valid_7_q,
			entry_payload_valid_8_i => ldq_addr_valid_8_q,
			entry_payload_valid_9_i => ldq_addr_valid_9_q,
			entry_payload_0_o => ldq_addr_0_d,
			entry_payload_1_o => ldq_addr_1_d,
			entry_payload_2_o => ldq_addr_2_d,
			entry_payload_3_o => ldq_addr_3_d,
			entry_payload_4_o => ldq_addr_4_d,
			entry_payload_5_o => ldq_addr_5_d,
			entry_payload_6_o => ldq_addr_6_d,
			entry_payload_7_o => ldq_addr_7_d,
			entry_payload_8_o => ldq_addr_8_d,
			entry_payload_9_o => ldq_addr_9_d,
			entry_wen_0_o => ldq_addr_wen_0,
			entry_wen_1_o => ldq_addr_wen_1,
			entry_wen_2_o => ldq_addr_wen_2,
			entry_wen_3_o => ldq_addr_wen_3,
			entry_wen_4_o => ldq_addr_wen_4,
			entry_wen_5_o => ldq_addr_wen_5,
			entry_wen_6_o => ldq_addr_wen_6,
			entry_wen_7_o => ldq_addr_wen_7,
			entry_wen_8_o => ldq_addr_wen_8,
			entry_wen_9_o => ldq_addr_wen_9,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq1_core_ldd_dispatcher : entity work.handshake_lsq_lsq1_core_ldd
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_o => ldp_data_0_o,
			port_ready_0_i => ldp_data_ready_0_i,
			port_valid_0_o => ldp_data_valid_0_o,
			entry_alloc_0_i => ldq_alloc_0_q,
			entry_alloc_1_i => ldq_alloc_1_q,
			entry_alloc_2_i => ldq_alloc_2_q,
			entry_alloc_3_i => ldq_alloc_3_q,
			entry_alloc_4_i => ldq_alloc_4_q,
			entry_alloc_5_i => ldq_alloc_5_q,
			entry_alloc_6_i => ldq_alloc_6_q,
			entry_alloc_7_i => ldq_alloc_7_q,
			entry_alloc_8_i => ldq_alloc_8_q,
			entry_alloc_9_i => ldq_alloc_9_q,
			entry_payload_valid_0_i => ldq_data_valid_0_q,
			entry_payload_valid_1_i => ldq_data_valid_1_q,
			entry_payload_valid_2_i => ldq_data_valid_2_q,
			entry_payload_valid_3_i => ldq_data_valid_3_q,
			entry_payload_valid_4_i => ldq_data_valid_4_q,
			entry_payload_valid_5_i => ldq_data_valid_5_q,
			entry_payload_valid_6_i => ldq_data_valid_6_q,
			entry_payload_valid_7_i => ldq_data_valid_7_q,
			entry_payload_valid_8_i => ldq_data_valid_8_q,
			entry_payload_valid_9_i => ldq_data_valid_9_q,
			entry_payload_0_i => ldq_data_0_q,
			entry_payload_1_i => ldq_data_1_q,
			entry_payload_2_i => ldq_data_2_q,
			entry_payload_3_i => ldq_data_3_q,
			entry_payload_4_i => ldq_data_4_q,
			entry_payload_5_i => ldq_data_5_q,
			entry_payload_6_i => ldq_data_6_q,
			entry_payload_7_i => ldq_data_7_q,
			entry_payload_8_i => ldq_data_8_q,
			entry_payload_9_i => ldq_data_9_q,
			entry_reset_0_o => ldq_reset_0,
			entry_reset_1_o => ldq_reset_1,
			entry_reset_2_o => ldq_reset_2,
			entry_reset_3_o => ldq_reset_3,
			entry_reset_4_o => ldq_reset_4,
			entry_reset_5_o => ldq_reset_5,
			entry_reset_6_o => ldq_reset_6,
			entry_reset_7_o => ldq_reset_7,
			entry_reset_8_o => ldq_reset_8,
			entry_reset_9_o => ldq_reset_9,
			queue_head_oh_i => ldq_head_oh
		);
	handshake_lsq_lsq1_core_sta_dispatcher : entity work.handshake_lsq_lsq1_core_sta
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_addr_0_i,
			port_ready_0_o => stp_addr_ready_0_o,
			port_valid_0_i => stp_addr_valid_0_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_alloc_8_i => stq_alloc_8_q,
			entry_alloc_9_i => stq_alloc_9_q,
			entry_payload_valid_0_i => stq_addr_valid_0_q,
			entry_payload_valid_1_i => stq_addr_valid_1_q,
			entry_payload_valid_2_i => stq_addr_valid_2_q,
			entry_payload_valid_3_i => stq_addr_valid_3_q,
			entry_payload_valid_4_i => stq_addr_valid_4_q,
			entry_payload_valid_5_i => stq_addr_valid_5_q,
			entry_payload_valid_6_i => stq_addr_valid_6_q,
			entry_payload_valid_7_i => stq_addr_valid_7_q,
			entry_payload_valid_8_i => stq_addr_valid_8_q,
			entry_payload_valid_9_i => stq_addr_valid_9_q,
			entry_payload_0_o => stq_addr_0_d,
			entry_payload_1_o => stq_addr_1_d,
			entry_payload_2_o => stq_addr_2_d,
			entry_payload_3_o => stq_addr_3_d,
			entry_payload_4_o => stq_addr_4_d,
			entry_payload_5_o => stq_addr_5_d,
			entry_payload_6_o => stq_addr_6_d,
			entry_payload_7_o => stq_addr_7_d,
			entry_payload_8_o => stq_addr_8_d,
			entry_payload_9_o => stq_addr_9_d,
			entry_wen_0_o => stq_addr_wen_0,
			entry_wen_1_o => stq_addr_wen_1,
			entry_wen_2_o => stq_addr_wen_2,
			entry_wen_3_o => stq_addr_wen_3,
			entry_wen_4_o => stq_addr_wen_4,
			entry_wen_5_o => stq_addr_wen_5,
			entry_wen_6_o => stq_addr_wen_6,
			entry_wen_7_o => stq_addr_wen_7,
			entry_wen_8_o => stq_addr_wen_8,
			entry_wen_9_o => stq_addr_wen_9,
			queue_head_oh_i => stq_head_oh
		);
	handshake_lsq_lsq1_core_std_dispatcher : entity work.handshake_lsq_lsq1_core_std
		port map(
			rst => rst,
			clk => clk,
			port_payload_0_i => stp_data_0_i,
			port_ready_0_o => stp_data_ready_0_o,
			port_valid_0_i => stp_data_valid_0_i,
			entry_alloc_0_i => stq_alloc_0_q,
			entry_alloc_1_i => stq_alloc_1_q,
			entry_alloc_2_i => stq_alloc_2_q,
			entry_alloc_3_i => stq_alloc_3_q,
			entry_alloc_4_i => stq_alloc_4_q,
			entry_alloc_5_i => stq_alloc_5_q,
			entry_alloc_6_i => stq_alloc_6_q,
			entry_alloc_7_i => stq_alloc_7_q,
			entry_alloc_8_i => stq_alloc_8_q,
			entry_alloc_9_i => stq_alloc_9_q,
			entry_payload_valid_0_i => stq_data_valid_0_q,
			entry_payload_valid_1_i => stq_data_valid_1_q,
			entry_payload_valid_2_i => stq_data_valid_2_q,
			entry_payload_valid_3_i => stq_data_valid_3_q,
			entry_payload_valid_4_i => stq_data_valid_4_q,
			entry_payload_valid_5_i => stq_data_valid_5_q,
			entry_payload_valid_6_i => stq_data_valid_6_q,
			entry_payload_valid_7_i => stq_data_valid_7_q,
			entry_payload_valid_8_i => stq_data_valid_8_q,
			entry_payload_valid_9_i => stq_data_valid_9_q,
			entry_payload_0_o => stq_data_0_d,
			entry_payload_1_o => stq_data_1_d,
			entry_payload_2_o => stq_data_2_d,
			entry_payload_3_o => stq_data_3_d,
			entry_payload_4_o => stq_data_4_d,
			entry_payload_5_o => stq_data_5_d,
			entry_payload_6_o => stq_data_6_d,
			entry_payload_7_o => stq_data_7_d,
			entry_payload_8_o => stq_data_8_d,
			entry_payload_9_o => stq_data_9_d,
			entry_wen_0_o => stq_data_wen_0,
			entry_wen_1_o => stq_data_wen_1,
			entry_wen_2_o => stq_data_wen_2,
			entry_wen_3_o => stq_data_wen_3,
			entry_wen_4_o => stq_data_wen_4,
			entry_wen_5_o => stq_data_wen_5,
			entry_wen_6_o => stq_data_wen_6,
			entry_wen_7_o => stq_data_wen_7,
			entry_wen_8_o => stq_data_wen_8,
			entry_wen_9_o => stq_data_wen_9,
			queue_head_oh_i => stq_head_oh
		);
	addr_valid_0(0) <= ldq_addr_valid_0_q and stq_addr_valid_0_q;
	addr_valid_0(1) <= ldq_addr_valid_0_q and stq_addr_valid_1_q;
	addr_valid_0(2) <= ldq_addr_valid_0_q and stq_addr_valid_2_q;
	addr_valid_0(3) <= ldq_addr_valid_0_q and stq_addr_valid_3_q;
	addr_valid_0(4) <= ldq_addr_valid_0_q and stq_addr_valid_4_q;
	addr_valid_0(5) <= ldq_addr_valid_0_q and stq_addr_valid_5_q;
	addr_valid_0(6) <= ldq_addr_valid_0_q and stq_addr_valid_6_q;
	addr_valid_0(7) <= ldq_addr_valid_0_q and stq_addr_valid_7_q;
	addr_valid_0(8) <= ldq_addr_valid_0_q and stq_addr_valid_8_q;
	addr_valid_0(9) <= ldq_addr_valid_0_q and stq_addr_valid_9_q;
	addr_valid_1(0) <= ldq_addr_valid_1_q and stq_addr_valid_0_q;
	addr_valid_1(1) <= ldq_addr_valid_1_q and stq_addr_valid_1_q;
	addr_valid_1(2) <= ldq_addr_valid_1_q and stq_addr_valid_2_q;
	addr_valid_1(3) <= ldq_addr_valid_1_q and stq_addr_valid_3_q;
	addr_valid_1(4) <= ldq_addr_valid_1_q and stq_addr_valid_4_q;
	addr_valid_1(5) <= ldq_addr_valid_1_q and stq_addr_valid_5_q;
	addr_valid_1(6) <= ldq_addr_valid_1_q and stq_addr_valid_6_q;
	addr_valid_1(7) <= ldq_addr_valid_1_q and stq_addr_valid_7_q;
	addr_valid_1(8) <= ldq_addr_valid_1_q and stq_addr_valid_8_q;
	addr_valid_1(9) <= ldq_addr_valid_1_q and stq_addr_valid_9_q;
	addr_valid_2(0) <= ldq_addr_valid_2_q and stq_addr_valid_0_q;
	addr_valid_2(1) <= ldq_addr_valid_2_q and stq_addr_valid_1_q;
	addr_valid_2(2) <= ldq_addr_valid_2_q and stq_addr_valid_2_q;
	addr_valid_2(3) <= ldq_addr_valid_2_q and stq_addr_valid_3_q;
	addr_valid_2(4) <= ldq_addr_valid_2_q and stq_addr_valid_4_q;
	addr_valid_2(5) <= ldq_addr_valid_2_q and stq_addr_valid_5_q;
	addr_valid_2(6) <= ldq_addr_valid_2_q and stq_addr_valid_6_q;
	addr_valid_2(7) <= ldq_addr_valid_2_q and stq_addr_valid_7_q;
	addr_valid_2(8) <= ldq_addr_valid_2_q and stq_addr_valid_8_q;
	addr_valid_2(9) <= ldq_addr_valid_2_q and stq_addr_valid_9_q;
	addr_valid_3(0) <= ldq_addr_valid_3_q and stq_addr_valid_0_q;
	addr_valid_3(1) <= ldq_addr_valid_3_q and stq_addr_valid_1_q;
	addr_valid_3(2) <= ldq_addr_valid_3_q and stq_addr_valid_2_q;
	addr_valid_3(3) <= ldq_addr_valid_3_q and stq_addr_valid_3_q;
	addr_valid_3(4) <= ldq_addr_valid_3_q and stq_addr_valid_4_q;
	addr_valid_3(5) <= ldq_addr_valid_3_q and stq_addr_valid_5_q;
	addr_valid_3(6) <= ldq_addr_valid_3_q and stq_addr_valid_6_q;
	addr_valid_3(7) <= ldq_addr_valid_3_q and stq_addr_valid_7_q;
	addr_valid_3(8) <= ldq_addr_valid_3_q and stq_addr_valid_8_q;
	addr_valid_3(9) <= ldq_addr_valid_3_q and stq_addr_valid_9_q;
	addr_valid_4(0) <= ldq_addr_valid_4_q and stq_addr_valid_0_q;
	addr_valid_4(1) <= ldq_addr_valid_4_q and stq_addr_valid_1_q;
	addr_valid_4(2) <= ldq_addr_valid_4_q and stq_addr_valid_2_q;
	addr_valid_4(3) <= ldq_addr_valid_4_q and stq_addr_valid_3_q;
	addr_valid_4(4) <= ldq_addr_valid_4_q and stq_addr_valid_4_q;
	addr_valid_4(5) <= ldq_addr_valid_4_q and stq_addr_valid_5_q;
	addr_valid_4(6) <= ldq_addr_valid_4_q and stq_addr_valid_6_q;
	addr_valid_4(7) <= ldq_addr_valid_4_q and stq_addr_valid_7_q;
	addr_valid_4(8) <= ldq_addr_valid_4_q and stq_addr_valid_8_q;
	addr_valid_4(9) <= ldq_addr_valid_4_q and stq_addr_valid_9_q;
	addr_valid_5(0) <= ldq_addr_valid_5_q and stq_addr_valid_0_q;
	addr_valid_5(1) <= ldq_addr_valid_5_q and stq_addr_valid_1_q;
	addr_valid_5(2) <= ldq_addr_valid_5_q and stq_addr_valid_2_q;
	addr_valid_5(3) <= ldq_addr_valid_5_q and stq_addr_valid_3_q;
	addr_valid_5(4) <= ldq_addr_valid_5_q and stq_addr_valid_4_q;
	addr_valid_5(5) <= ldq_addr_valid_5_q and stq_addr_valid_5_q;
	addr_valid_5(6) <= ldq_addr_valid_5_q and stq_addr_valid_6_q;
	addr_valid_5(7) <= ldq_addr_valid_5_q and stq_addr_valid_7_q;
	addr_valid_5(8) <= ldq_addr_valid_5_q and stq_addr_valid_8_q;
	addr_valid_5(9) <= ldq_addr_valid_5_q and stq_addr_valid_9_q;
	addr_valid_6(0) <= ldq_addr_valid_6_q and stq_addr_valid_0_q;
	addr_valid_6(1) <= ldq_addr_valid_6_q and stq_addr_valid_1_q;
	addr_valid_6(2) <= ldq_addr_valid_6_q and stq_addr_valid_2_q;
	addr_valid_6(3) <= ldq_addr_valid_6_q and stq_addr_valid_3_q;
	addr_valid_6(4) <= ldq_addr_valid_6_q and stq_addr_valid_4_q;
	addr_valid_6(5) <= ldq_addr_valid_6_q and stq_addr_valid_5_q;
	addr_valid_6(6) <= ldq_addr_valid_6_q and stq_addr_valid_6_q;
	addr_valid_6(7) <= ldq_addr_valid_6_q and stq_addr_valid_7_q;
	addr_valid_6(8) <= ldq_addr_valid_6_q and stq_addr_valid_8_q;
	addr_valid_6(9) <= ldq_addr_valid_6_q and stq_addr_valid_9_q;
	addr_valid_7(0) <= ldq_addr_valid_7_q and stq_addr_valid_0_q;
	addr_valid_7(1) <= ldq_addr_valid_7_q and stq_addr_valid_1_q;
	addr_valid_7(2) <= ldq_addr_valid_7_q and stq_addr_valid_2_q;
	addr_valid_7(3) <= ldq_addr_valid_7_q and stq_addr_valid_3_q;
	addr_valid_7(4) <= ldq_addr_valid_7_q and stq_addr_valid_4_q;
	addr_valid_7(5) <= ldq_addr_valid_7_q and stq_addr_valid_5_q;
	addr_valid_7(6) <= ldq_addr_valid_7_q and stq_addr_valid_6_q;
	addr_valid_7(7) <= ldq_addr_valid_7_q and stq_addr_valid_7_q;
	addr_valid_7(8) <= ldq_addr_valid_7_q and stq_addr_valid_8_q;
	addr_valid_7(9) <= ldq_addr_valid_7_q and stq_addr_valid_9_q;
	addr_valid_8(0) <= ldq_addr_valid_8_q and stq_addr_valid_0_q;
	addr_valid_8(1) <= ldq_addr_valid_8_q and stq_addr_valid_1_q;
	addr_valid_8(2) <= ldq_addr_valid_8_q and stq_addr_valid_2_q;
	addr_valid_8(3) <= ldq_addr_valid_8_q and stq_addr_valid_3_q;
	addr_valid_8(4) <= ldq_addr_valid_8_q and stq_addr_valid_4_q;
	addr_valid_8(5) <= ldq_addr_valid_8_q and stq_addr_valid_5_q;
	addr_valid_8(6) <= ldq_addr_valid_8_q and stq_addr_valid_6_q;
	addr_valid_8(7) <= ldq_addr_valid_8_q and stq_addr_valid_7_q;
	addr_valid_8(8) <= ldq_addr_valid_8_q and stq_addr_valid_8_q;
	addr_valid_8(9) <= ldq_addr_valid_8_q and stq_addr_valid_9_q;
	addr_valid_9(0) <= ldq_addr_valid_9_q and stq_addr_valid_0_q;
	addr_valid_9(1) <= ldq_addr_valid_9_q and stq_addr_valid_1_q;
	addr_valid_9(2) <= ldq_addr_valid_9_q and stq_addr_valid_2_q;
	addr_valid_9(3) <= ldq_addr_valid_9_q and stq_addr_valid_3_q;
	addr_valid_9(4) <= ldq_addr_valid_9_q and stq_addr_valid_4_q;
	addr_valid_9(5) <= ldq_addr_valid_9_q and stq_addr_valid_5_q;
	addr_valid_9(6) <= ldq_addr_valid_9_q and stq_addr_valid_6_q;
	addr_valid_9(7) <= ldq_addr_valid_9_q and stq_addr_valid_7_q;
	addr_valid_9(8) <= ldq_addr_valid_9_q and stq_addr_valid_8_q;
	addr_valid_9(9) <= ldq_addr_valid_9_q and stq_addr_valid_9_q;
	addr_same_0(0) <= '1' when ldq_addr_0_q = stq_addr_0_q else '0';
	addr_same_0(1) <= '1' when ldq_addr_0_q = stq_addr_1_q else '0';
	addr_same_0(2) <= '1' when ldq_addr_0_q = stq_addr_2_q else '0';
	addr_same_0(3) <= '1' when ldq_addr_0_q = stq_addr_3_q else '0';
	addr_same_0(4) <= '1' when ldq_addr_0_q = stq_addr_4_q else '0';
	addr_same_0(5) <= '1' when ldq_addr_0_q = stq_addr_5_q else '0';
	addr_same_0(6) <= '1' when ldq_addr_0_q = stq_addr_6_q else '0';
	addr_same_0(7) <= '1' when ldq_addr_0_q = stq_addr_7_q else '0';
	addr_same_0(8) <= '1' when ldq_addr_0_q = stq_addr_8_q else '0';
	addr_same_0(9) <= '1' when ldq_addr_0_q = stq_addr_9_q else '0';
	addr_same_1(0) <= '1' when ldq_addr_1_q = stq_addr_0_q else '0';
	addr_same_1(1) <= '1' when ldq_addr_1_q = stq_addr_1_q else '0';
	addr_same_1(2) <= '1' when ldq_addr_1_q = stq_addr_2_q else '0';
	addr_same_1(3) <= '1' when ldq_addr_1_q = stq_addr_3_q else '0';
	addr_same_1(4) <= '1' when ldq_addr_1_q = stq_addr_4_q else '0';
	addr_same_1(5) <= '1' when ldq_addr_1_q = stq_addr_5_q else '0';
	addr_same_1(6) <= '1' when ldq_addr_1_q = stq_addr_6_q else '0';
	addr_same_1(7) <= '1' when ldq_addr_1_q = stq_addr_7_q else '0';
	addr_same_1(8) <= '1' when ldq_addr_1_q = stq_addr_8_q else '0';
	addr_same_1(9) <= '1' when ldq_addr_1_q = stq_addr_9_q else '0';
	addr_same_2(0) <= '1' when ldq_addr_2_q = stq_addr_0_q else '0';
	addr_same_2(1) <= '1' when ldq_addr_2_q = stq_addr_1_q else '0';
	addr_same_2(2) <= '1' when ldq_addr_2_q = stq_addr_2_q else '0';
	addr_same_2(3) <= '1' when ldq_addr_2_q = stq_addr_3_q else '0';
	addr_same_2(4) <= '1' when ldq_addr_2_q = stq_addr_4_q else '0';
	addr_same_2(5) <= '1' when ldq_addr_2_q = stq_addr_5_q else '0';
	addr_same_2(6) <= '1' when ldq_addr_2_q = stq_addr_6_q else '0';
	addr_same_2(7) <= '1' when ldq_addr_2_q = stq_addr_7_q else '0';
	addr_same_2(8) <= '1' when ldq_addr_2_q = stq_addr_8_q else '0';
	addr_same_2(9) <= '1' when ldq_addr_2_q = stq_addr_9_q else '0';
	addr_same_3(0) <= '1' when ldq_addr_3_q = stq_addr_0_q else '0';
	addr_same_3(1) <= '1' when ldq_addr_3_q = stq_addr_1_q else '0';
	addr_same_3(2) <= '1' when ldq_addr_3_q = stq_addr_2_q else '0';
	addr_same_3(3) <= '1' when ldq_addr_3_q = stq_addr_3_q else '0';
	addr_same_3(4) <= '1' when ldq_addr_3_q = stq_addr_4_q else '0';
	addr_same_3(5) <= '1' when ldq_addr_3_q = stq_addr_5_q else '0';
	addr_same_3(6) <= '1' when ldq_addr_3_q = stq_addr_6_q else '0';
	addr_same_3(7) <= '1' when ldq_addr_3_q = stq_addr_7_q else '0';
	addr_same_3(8) <= '1' when ldq_addr_3_q = stq_addr_8_q else '0';
	addr_same_3(9) <= '1' when ldq_addr_3_q = stq_addr_9_q else '0';
	addr_same_4(0) <= '1' when ldq_addr_4_q = stq_addr_0_q else '0';
	addr_same_4(1) <= '1' when ldq_addr_4_q = stq_addr_1_q else '0';
	addr_same_4(2) <= '1' when ldq_addr_4_q = stq_addr_2_q else '0';
	addr_same_4(3) <= '1' when ldq_addr_4_q = stq_addr_3_q else '0';
	addr_same_4(4) <= '1' when ldq_addr_4_q = stq_addr_4_q else '0';
	addr_same_4(5) <= '1' when ldq_addr_4_q = stq_addr_5_q else '0';
	addr_same_4(6) <= '1' when ldq_addr_4_q = stq_addr_6_q else '0';
	addr_same_4(7) <= '1' when ldq_addr_4_q = stq_addr_7_q else '0';
	addr_same_4(8) <= '1' when ldq_addr_4_q = stq_addr_8_q else '0';
	addr_same_4(9) <= '1' when ldq_addr_4_q = stq_addr_9_q else '0';
	addr_same_5(0) <= '1' when ldq_addr_5_q = stq_addr_0_q else '0';
	addr_same_5(1) <= '1' when ldq_addr_5_q = stq_addr_1_q else '0';
	addr_same_5(2) <= '1' when ldq_addr_5_q = stq_addr_2_q else '0';
	addr_same_5(3) <= '1' when ldq_addr_5_q = stq_addr_3_q else '0';
	addr_same_5(4) <= '1' when ldq_addr_5_q = stq_addr_4_q else '0';
	addr_same_5(5) <= '1' when ldq_addr_5_q = stq_addr_5_q else '0';
	addr_same_5(6) <= '1' when ldq_addr_5_q = stq_addr_6_q else '0';
	addr_same_5(7) <= '1' when ldq_addr_5_q = stq_addr_7_q else '0';
	addr_same_5(8) <= '1' when ldq_addr_5_q = stq_addr_8_q else '0';
	addr_same_5(9) <= '1' when ldq_addr_5_q = stq_addr_9_q else '0';
	addr_same_6(0) <= '1' when ldq_addr_6_q = stq_addr_0_q else '0';
	addr_same_6(1) <= '1' when ldq_addr_6_q = stq_addr_1_q else '0';
	addr_same_6(2) <= '1' when ldq_addr_6_q = stq_addr_2_q else '0';
	addr_same_6(3) <= '1' when ldq_addr_6_q = stq_addr_3_q else '0';
	addr_same_6(4) <= '1' when ldq_addr_6_q = stq_addr_4_q else '0';
	addr_same_6(5) <= '1' when ldq_addr_6_q = stq_addr_5_q else '0';
	addr_same_6(6) <= '1' when ldq_addr_6_q = stq_addr_6_q else '0';
	addr_same_6(7) <= '1' when ldq_addr_6_q = stq_addr_7_q else '0';
	addr_same_6(8) <= '1' when ldq_addr_6_q = stq_addr_8_q else '0';
	addr_same_6(9) <= '1' when ldq_addr_6_q = stq_addr_9_q else '0';
	addr_same_7(0) <= '1' when ldq_addr_7_q = stq_addr_0_q else '0';
	addr_same_7(1) <= '1' when ldq_addr_7_q = stq_addr_1_q else '0';
	addr_same_7(2) <= '1' when ldq_addr_7_q = stq_addr_2_q else '0';
	addr_same_7(3) <= '1' when ldq_addr_7_q = stq_addr_3_q else '0';
	addr_same_7(4) <= '1' when ldq_addr_7_q = stq_addr_4_q else '0';
	addr_same_7(5) <= '1' when ldq_addr_7_q = stq_addr_5_q else '0';
	addr_same_7(6) <= '1' when ldq_addr_7_q = stq_addr_6_q else '0';
	addr_same_7(7) <= '1' when ldq_addr_7_q = stq_addr_7_q else '0';
	addr_same_7(8) <= '1' when ldq_addr_7_q = stq_addr_8_q else '0';
	addr_same_7(9) <= '1' when ldq_addr_7_q = stq_addr_9_q else '0';
	addr_same_8(0) <= '1' when ldq_addr_8_q = stq_addr_0_q else '0';
	addr_same_8(1) <= '1' when ldq_addr_8_q = stq_addr_1_q else '0';
	addr_same_8(2) <= '1' when ldq_addr_8_q = stq_addr_2_q else '0';
	addr_same_8(3) <= '1' when ldq_addr_8_q = stq_addr_3_q else '0';
	addr_same_8(4) <= '1' when ldq_addr_8_q = stq_addr_4_q else '0';
	addr_same_8(5) <= '1' when ldq_addr_8_q = stq_addr_5_q else '0';
	addr_same_8(6) <= '1' when ldq_addr_8_q = stq_addr_6_q else '0';
	addr_same_8(7) <= '1' when ldq_addr_8_q = stq_addr_7_q else '0';
	addr_same_8(8) <= '1' when ldq_addr_8_q = stq_addr_8_q else '0';
	addr_same_8(9) <= '1' when ldq_addr_8_q = stq_addr_9_q else '0';
	addr_same_9(0) <= '1' when ldq_addr_9_q = stq_addr_0_q else '0';
	addr_same_9(1) <= '1' when ldq_addr_9_q = stq_addr_1_q else '0';
	addr_same_9(2) <= '1' when ldq_addr_9_q = stq_addr_2_q else '0';
	addr_same_9(3) <= '1' when ldq_addr_9_q = stq_addr_3_q else '0';
	addr_same_9(4) <= '1' when ldq_addr_9_q = stq_addr_4_q else '0';
	addr_same_9(5) <= '1' when ldq_addr_9_q = stq_addr_5_q else '0';
	addr_same_9(6) <= '1' when ldq_addr_9_q = stq_addr_6_q else '0';
	addr_same_9(7) <= '1' when ldq_addr_9_q = stq_addr_7_q else '0';
	addr_same_9(8) <= '1' when ldq_addr_9_q = stq_addr_8_q else '0';
	addr_same_9(9) <= '1' when ldq_addr_9_q = stq_addr_9_q else '0';
	ld_st_conflict_0(0) <= stq_alloc_0_q and store_is_older_0_q(0) and ( addr_same_0(0) or not stq_addr_valid_0_q );
	ld_st_conflict_0(1) <= stq_alloc_1_q and store_is_older_0_q(1) and ( addr_same_0(1) or not stq_addr_valid_1_q );
	ld_st_conflict_0(2) <= stq_alloc_2_q and store_is_older_0_q(2) and ( addr_same_0(2) or not stq_addr_valid_2_q );
	ld_st_conflict_0(3) <= stq_alloc_3_q and store_is_older_0_q(3) and ( addr_same_0(3) or not stq_addr_valid_3_q );
	ld_st_conflict_0(4) <= stq_alloc_4_q and store_is_older_0_q(4) and ( addr_same_0(4) or not stq_addr_valid_4_q );
	ld_st_conflict_0(5) <= stq_alloc_5_q and store_is_older_0_q(5) and ( addr_same_0(5) or not stq_addr_valid_5_q );
	ld_st_conflict_0(6) <= stq_alloc_6_q and store_is_older_0_q(6) and ( addr_same_0(6) or not stq_addr_valid_6_q );
	ld_st_conflict_0(7) <= stq_alloc_7_q and store_is_older_0_q(7) and ( addr_same_0(7) or not stq_addr_valid_7_q );
	ld_st_conflict_0(8) <= stq_alloc_8_q and store_is_older_0_q(8) and ( addr_same_0(8) or not stq_addr_valid_8_q );
	ld_st_conflict_0(9) <= stq_alloc_9_q and store_is_older_0_q(9) and ( addr_same_0(9) or not stq_addr_valid_9_q );
	ld_st_conflict_1(0) <= stq_alloc_0_q and store_is_older_1_q(0) and ( addr_same_1(0) or not stq_addr_valid_0_q );
	ld_st_conflict_1(1) <= stq_alloc_1_q and store_is_older_1_q(1) and ( addr_same_1(1) or not stq_addr_valid_1_q );
	ld_st_conflict_1(2) <= stq_alloc_2_q and store_is_older_1_q(2) and ( addr_same_1(2) or not stq_addr_valid_2_q );
	ld_st_conflict_1(3) <= stq_alloc_3_q and store_is_older_1_q(3) and ( addr_same_1(3) or not stq_addr_valid_3_q );
	ld_st_conflict_1(4) <= stq_alloc_4_q and store_is_older_1_q(4) and ( addr_same_1(4) or not stq_addr_valid_4_q );
	ld_st_conflict_1(5) <= stq_alloc_5_q and store_is_older_1_q(5) and ( addr_same_1(5) or not stq_addr_valid_5_q );
	ld_st_conflict_1(6) <= stq_alloc_6_q and store_is_older_1_q(6) and ( addr_same_1(6) or not stq_addr_valid_6_q );
	ld_st_conflict_1(7) <= stq_alloc_7_q and store_is_older_1_q(7) and ( addr_same_1(7) or not stq_addr_valid_7_q );
	ld_st_conflict_1(8) <= stq_alloc_8_q and store_is_older_1_q(8) and ( addr_same_1(8) or not stq_addr_valid_8_q );
	ld_st_conflict_1(9) <= stq_alloc_9_q and store_is_older_1_q(9) and ( addr_same_1(9) or not stq_addr_valid_9_q );
	ld_st_conflict_2(0) <= stq_alloc_0_q and store_is_older_2_q(0) and ( addr_same_2(0) or not stq_addr_valid_0_q );
	ld_st_conflict_2(1) <= stq_alloc_1_q and store_is_older_2_q(1) and ( addr_same_2(1) or not stq_addr_valid_1_q );
	ld_st_conflict_2(2) <= stq_alloc_2_q and store_is_older_2_q(2) and ( addr_same_2(2) or not stq_addr_valid_2_q );
	ld_st_conflict_2(3) <= stq_alloc_3_q and store_is_older_2_q(3) and ( addr_same_2(3) or not stq_addr_valid_3_q );
	ld_st_conflict_2(4) <= stq_alloc_4_q and store_is_older_2_q(4) and ( addr_same_2(4) or not stq_addr_valid_4_q );
	ld_st_conflict_2(5) <= stq_alloc_5_q and store_is_older_2_q(5) and ( addr_same_2(5) or not stq_addr_valid_5_q );
	ld_st_conflict_2(6) <= stq_alloc_6_q and store_is_older_2_q(6) and ( addr_same_2(6) or not stq_addr_valid_6_q );
	ld_st_conflict_2(7) <= stq_alloc_7_q and store_is_older_2_q(7) and ( addr_same_2(7) or not stq_addr_valid_7_q );
	ld_st_conflict_2(8) <= stq_alloc_8_q and store_is_older_2_q(8) and ( addr_same_2(8) or not stq_addr_valid_8_q );
	ld_st_conflict_2(9) <= stq_alloc_9_q and store_is_older_2_q(9) and ( addr_same_2(9) or not stq_addr_valid_9_q );
	ld_st_conflict_3(0) <= stq_alloc_0_q and store_is_older_3_q(0) and ( addr_same_3(0) or not stq_addr_valid_0_q );
	ld_st_conflict_3(1) <= stq_alloc_1_q and store_is_older_3_q(1) and ( addr_same_3(1) or not stq_addr_valid_1_q );
	ld_st_conflict_3(2) <= stq_alloc_2_q and store_is_older_3_q(2) and ( addr_same_3(2) or not stq_addr_valid_2_q );
	ld_st_conflict_3(3) <= stq_alloc_3_q and store_is_older_3_q(3) and ( addr_same_3(3) or not stq_addr_valid_3_q );
	ld_st_conflict_3(4) <= stq_alloc_4_q and store_is_older_3_q(4) and ( addr_same_3(4) or not stq_addr_valid_4_q );
	ld_st_conflict_3(5) <= stq_alloc_5_q and store_is_older_3_q(5) and ( addr_same_3(5) or not stq_addr_valid_5_q );
	ld_st_conflict_3(6) <= stq_alloc_6_q and store_is_older_3_q(6) and ( addr_same_3(6) or not stq_addr_valid_6_q );
	ld_st_conflict_3(7) <= stq_alloc_7_q and store_is_older_3_q(7) and ( addr_same_3(7) or not stq_addr_valid_7_q );
	ld_st_conflict_3(8) <= stq_alloc_8_q and store_is_older_3_q(8) and ( addr_same_3(8) or not stq_addr_valid_8_q );
	ld_st_conflict_3(9) <= stq_alloc_9_q and store_is_older_3_q(9) and ( addr_same_3(9) or not stq_addr_valid_9_q );
	ld_st_conflict_4(0) <= stq_alloc_0_q and store_is_older_4_q(0) and ( addr_same_4(0) or not stq_addr_valid_0_q );
	ld_st_conflict_4(1) <= stq_alloc_1_q and store_is_older_4_q(1) and ( addr_same_4(1) or not stq_addr_valid_1_q );
	ld_st_conflict_4(2) <= stq_alloc_2_q and store_is_older_4_q(2) and ( addr_same_4(2) or not stq_addr_valid_2_q );
	ld_st_conflict_4(3) <= stq_alloc_3_q and store_is_older_4_q(3) and ( addr_same_4(3) or not stq_addr_valid_3_q );
	ld_st_conflict_4(4) <= stq_alloc_4_q and store_is_older_4_q(4) and ( addr_same_4(4) or not stq_addr_valid_4_q );
	ld_st_conflict_4(5) <= stq_alloc_5_q and store_is_older_4_q(5) and ( addr_same_4(5) or not stq_addr_valid_5_q );
	ld_st_conflict_4(6) <= stq_alloc_6_q and store_is_older_4_q(6) and ( addr_same_4(6) or not stq_addr_valid_6_q );
	ld_st_conflict_4(7) <= stq_alloc_7_q and store_is_older_4_q(7) and ( addr_same_4(7) or not stq_addr_valid_7_q );
	ld_st_conflict_4(8) <= stq_alloc_8_q and store_is_older_4_q(8) and ( addr_same_4(8) or not stq_addr_valid_8_q );
	ld_st_conflict_4(9) <= stq_alloc_9_q and store_is_older_4_q(9) and ( addr_same_4(9) or not stq_addr_valid_9_q );
	ld_st_conflict_5(0) <= stq_alloc_0_q and store_is_older_5_q(0) and ( addr_same_5(0) or not stq_addr_valid_0_q );
	ld_st_conflict_5(1) <= stq_alloc_1_q and store_is_older_5_q(1) and ( addr_same_5(1) or not stq_addr_valid_1_q );
	ld_st_conflict_5(2) <= stq_alloc_2_q and store_is_older_5_q(2) and ( addr_same_5(2) or not stq_addr_valid_2_q );
	ld_st_conflict_5(3) <= stq_alloc_3_q and store_is_older_5_q(3) and ( addr_same_5(3) or not stq_addr_valid_3_q );
	ld_st_conflict_5(4) <= stq_alloc_4_q and store_is_older_5_q(4) and ( addr_same_5(4) or not stq_addr_valid_4_q );
	ld_st_conflict_5(5) <= stq_alloc_5_q and store_is_older_5_q(5) and ( addr_same_5(5) or not stq_addr_valid_5_q );
	ld_st_conflict_5(6) <= stq_alloc_6_q and store_is_older_5_q(6) and ( addr_same_5(6) or not stq_addr_valid_6_q );
	ld_st_conflict_5(7) <= stq_alloc_7_q and store_is_older_5_q(7) and ( addr_same_5(7) or not stq_addr_valid_7_q );
	ld_st_conflict_5(8) <= stq_alloc_8_q and store_is_older_5_q(8) and ( addr_same_5(8) or not stq_addr_valid_8_q );
	ld_st_conflict_5(9) <= stq_alloc_9_q and store_is_older_5_q(9) and ( addr_same_5(9) or not stq_addr_valid_9_q );
	ld_st_conflict_6(0) <= stq_alloc_0_q and store_is_older_6_q(0) and ( addr_same_6(0) or not stq_addr_valid_0_q );
	ld_st_conflict_6(1) <= stq_alloc_1_q and store_is_older_6_q(1) and ( addr_same_6(1) or not stq_addr_valid_1_q );
	ld_st_conflict_6(2) <= stq_alloc_2_q and store_is_older_6_q(2) and ( addr_same_6(2) or not stq_addr_valid_2_q );
	ld_st_conflict_6(3) <= stq_alloc_3_q and store_is_older_6_q(3) and ( addr_same_6(3) or not stq_addr_valid_3_q );
	ld_st_conflict_6(4) <= stq_alloc_4_q and store_is_older_6_q(4) and ( addr_same_6(4) or not stq_addr_valid_4_q );
	ld_st_conflict_6(5) <= stq_alloc_5_q and store_is_older_6_q(5) and ( addr_same_6(5) or not stq_addr_valid_5_q );
	ld_st_conflict_6(6) <= stq_alloc_6_q and store_is_older_6_q(6) and ( addr_same_6(6) or not stq_addr_valid_6_q );
	ld_st_conflict_6(7) <= stq_alloc_7_q and store_is_older_6_q(7) and ( addr_same_6(7) or not stq_addr_valid_7_q );
	ld_st_conflict_6(8) <= stq_alloc_8_q and store_is_older_6_q(8) and ( addr_same_6(8) or not stq_addr_valid_8_q );
	ld_st_conflict_6(9) <= stq_alloc_9_q and store_is_older_6_q(9) and ( addr_same_6(9) or not stq_addr_valid_9_q );
	ld_st_conflict_7(0) <= stq_alloc_0_q and store_is_older_7_q(0) and ( addr_same_7(0) or not stq_addr_valid_0_q );
	ld_st_conflict_7(1) <= stq_alloc_1_q and store_is_older_7_q(1) and ( addr_same_7(1) or not stq_addr_valid_1_q );
	ld_st_conflict_7(2) <= stq_alloc_2_q and store_is_older_7_q(2) and ( addr_same_7(2) or not stq_addr_valid_2_q );
	ld_st_conflict_7(3) <= stq_alloc_3_q and store_is_older_7_q(3) and ( addr_same_7(3) or not stq_addr_valid_3_q );
	ld_st_conflict_7(4) <= stq_alloc_4_q and store_is_older_7_q(4) and ( addr_same_7(4) or not stq_addr_valid_4_q );
	ld_st_conflict_7(5) <= stq_alloc_5_q and store_is_older_7_q(5) and ( addr_same_7(5) or not stq_addr_valid_5_q );
	ld_st_conflict_7(6) <= stq_alloc_6_q and store_is_older_7_q(6) and ( addr_same_7(6) or not stq_addr_valid_6_q );
	ld_st_conflict_7(7) <= stq_alloc_7_q and store_is_older_7_q(7) and ( addr_same_7(7) or not stq_addr_valid_7_q );
	ld_st_conflict_7(8) <= stq_alloc_8_q and store_is_older_7_q(8) and ( addr_same_7(8) or not stq_addr_valid_8_q );
	ld_st_conflict_7(9) <= stq_alloc_9_q and store_is_older_7_q(9) and ( addr_same_7(9) or not stq_addr_valid_9_q );
	ld_st_conflict_8(0) <= stq_alloc_0_q and store_is_older_8_q(0) and ( addr_same_8(0) or not stq_addr_valid_0_q );
	ld_st_conflict_8(1) <= stq_alloc_1_q and store_is_older_8_q(1) and ( addr_same_8(1) or not stq_addr_valid_1_q );
	ld_st_conflict_8(2) <= stq_alloc_2_q and store_is_older_8_q(2) and ( addr_same_8(2) or not stq_addr_valid_2_q );
	ld_st_conflict_8(3) <= stq_alloc_3_q and store_is_older_8_q(3) and ( addr_same_8(3) or not stq_addr_valid_3_q );
	ld_st_conflict_8(4) <= stq_alloc_4_q and store_is_older_8_q(4) and ( addr_same_8(4) or not stq_addr_valid_4_q );
	ld_st_conflict_8(5) <= stq_alloc_5_q and store_is_older_8_q(5) and ( addr_same_8(5) or not stq_addr_valid_5_q );
	ld_st_conflict_8(6) <= stq_alloc_6_q and store_is_older_8_q(6) and ( addr_same_8(6) or not stq_addr_valid_6_q );
	ld_st_conflict_8(7) <= stq_alloc_7_q and store_is_older_8_q(7) and ( addr_same_8(7) or not stq_addr_valid_7_q );
	ld_st_conflict_8(8) <= stq_alloc_8_q and store_is_older_8_q(8) and ( addr_same_8(8) or not stq_addr_valid_8_q );
	ld_st_conflict_8(9) <= stq_alloc_9_q and store_is_older_8_q(9) and ( addr_same_8(9) or not stq_addr_valid_9_q );
	ld_st_conflict_9(0) <= stq_alloc_0_q and store_is_older_9_q(0) and ( addr_same_9(0) or not stq_addr_valid_0_q );
	ld_st_conflict_9(1) <= stq_alloc_1_q and store_is_older_9_q(1) and ( addr_same_9(1) or not stq_addr_valid_1_q );
	ld_st_conflict_9(2) <= stq_alloc_2_q and store_is_older_9_q(2) and ( addr_same_9(2) or not stq_addr_valid_2_q );
	ld_st_conflict_9(3) <= stq_alloc_3_q and store_is_older_9_q(3) and ( addr_same_9(3) or not stq_addr_valid_3_q );
	ld_st_conflict_9(4) <= stq_alloc_4_q and store_is_older_9_q(4) and ( addr_same_9(4) or not stq_addr_valid_4_q );
	ld_st_conflict_9(5) <= stq_alloc_5_q and store_is_older_9_q(5) and ( addr_same_9(5) or not stq_addr_valid_5_q );
	ld_st_conflict_9(6) <= stq_alloc_6_q and store_is_older_9_q(6) and ( addr_same_9(6) or not stq_addr_valid_6_q );
	ld_st_conflict_9(7) <= stq_alloc_7_q and store_is_older_9_q(7) and ( addr_same_9(7) or not stq_addr_valid_7_q );
	ld_st_conflict_9(8) <= stq_alloc_8_q and store_is_older_9_q(8) and ( addr_same_9(8) or not stq_addr_valid_8_q );
	ld_st_conflict_9(9) <= stq_alloc_9_q and store_is_older_9_q(9) and ( addr_same_9(9) or not stq_addr_valid_9_q );
	can_bypass_0(0) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_0_q and addr_same_0(0) and addr_valid_0(0);
	can_bypass_0(1) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_1_q and addr_same_0(1) and addr_valid_0(1);
	can_bypass_0(2) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_2_q and addr_same_0(2) and addr_valid_0(2);
	can_bypass_0(3) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_3_q and addr_same_0(3) and addr_valid_0(3);
	can_bypass_0(4) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_4_q and addr_same_0(4) and addr_valid_0(4);
	can_bypass_0(5) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_5_q and addr_same_0(5) and addr_valid_0(5);
	can_bypass_0(6) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_6_q and addr_same_0(6) and addr_valid_0(6);
	can_bypass_0(7) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_7_q and addr_same_0(7) and addr_valid_0(7);
	can_bypass_0(8) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_8_q and addr_same_0(8) and addr_valid_0(8);
	can_bypass_0(9) <= ldq_alloc_0_q and not ldq_issue_0_q and stq_data_valid_9_q and addr_same_0(9) and addr_valid_0(9);
	can_bypass_1(0) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_0_q and addr_same_1(0) and addr_valid_1(0);
	can_bypass_1(1) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_1_q and addr_same_1(1) and addr_valid_1(1);
	can_bypass_1(2) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_2_q and addr_same_1(2) and addr_valid_1(2);
	can_bypass_1(3) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_3_q and addr_same_1(3) and addr_valid_1(3);
	can_bypass_1(4) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_4_q and addr_same_1(4) and addr_valid_1(4);
	can_bypass_1(5) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_5_q and addr_same_1(5) and addr_valid_1(5);
	can_bypass_1(6) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_6_q and addr_same_1(6) and addr_valid_1(6);
	can_bypass_1(7) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_7_q and addr_same_1(7) and addr_valid_1(7);
	can_bypass_1(8) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_8_q and addr_same_1(8) and addr_valid_1(8);
	can_bypass_1(9) <= ldq_alloc_1_q and not ldq_issue_1_q and stq_data_valid_9_q and addr_same_1(9) and addr_valid_1(9);
	can_bypass_2(0) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_0_q and addr_same_2(0) and addr_valid_2(0);
	can_bypass_2(1) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_1_q and addr_same_2(1) and addr_valid_2(1);
	can_bypass_2(2) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_2_q and addr_same_2(2) and addr_valid_2(2);
	can_bypass_2(3) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_3_q and addr_same_2(3) and addr_valid_2(3);
	can_bypass_2(4) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_4_q and addr_same_2(4) and addr_valid_2(4);
	can_bypass_2(5) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_5_q and addr_same_2(5) and addr_valid_2(5);
	can_bypass_2(6) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_6_q and addr_same_2(6) and addr_valid_2(6);
	can_bypass_2(7) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_7_q and addr_same_2(7) and addr_valid_2(7);
	can_bypass_2(8) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_8_q and addr_same_2(8) and addr_valid_2(8);
	can_bypass_2(9) <= ldq_alloc_2_q and not ldq_issue_2_q and stq_data_valid_9_q and addr_same_2(9) and addr_valid_2(9);
	can_bypass_3(0) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_0_q and addr_same_3(0) and addr_valid_3(0);
	can_bypass_3(1) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_1_q and addr_same_3(1) and addr_valid_3(1);
	can_bypass_3(2) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_2_q and addr_same_3(2) and addr_valid_3(2);
	can_bypass_3(3) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_3_q and addr_same_3(3) and addr_valid_3(3);
	can_bypass_3(4) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_4_q and addr_same_3(4) and addr_valid_3(4);
	can_bypass_3(5) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_5_q and addr_same_3(5) and addr_valid_3(5);
	can_bypass_3(6) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_6_q and addr_same_3(6) and addr_valid_3(6);
	can_bypass_3(7) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_7_q and addr_same_3(7) and addr_valid_3(7);
	can_bypass_3(8) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_8_q and addr_same_3(8) and addr_valid_3(8);
	can_bypass_3(9) <= ldq_alloc_3_q and not ldq_issue_3_q and stq_data_valid_9_q and addr_same_3(9) and addr_valid_3(9);
	can_bypass_4(0) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_0_q and addr_same_4(0) and addr_valid_4(0);
	can_bypass_4(1) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_1_q and addr_same_4(1) and addr_valid_4(1);
	can_bypass_4(2) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_2_q and addr_same_4(2) and addr_valid_4(2);
	can_bypass_4(3) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_3_q and addr_same_4(3) and addr_valid_4(3);
	can_bypass_4(4) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_4_q and addr_same_4(4) and addr_valid_4(4);
	can_bypass_4(5) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_5_q and addr_same_4(5) and addr_valid_4(5);
	can_bypass_4(6) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_6_q and addr_same_4(6) and addr_valid_4(6);
	can_bypass_4(7) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_7_q and addr_same_4(7) and addr_valid_4(7);
	can_bypass_4(8) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_8_q and addr_same_4(8) and addr_valid_4(8);
	can_bypass_4(9) <= ldq_alloc_4_q and not ldq_issue_4_q and stq_data_valid_9_q and addr_same_4(9) and addr_valid_4(9);
	can_bypass_5(0) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_0_q and addr_same_5(0) and addr_valid_5(0);
	can_bypass_5(1) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_1_q and addr_same_5(1) and addr_valid_5(1);
	can_bypass_5(2) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_2_q and addr_same_5(2) and addr_valid_5(2);
	can_bypass_5(3) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_3_q and addr_same_5(3) and addr_valid_5(3);
	can_bypass_5(4) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_4_q and addr_same_5(4) and addr_valid_5(4);
	can_bypass_5(5) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_5_q and addr_same_5(5) and addr_valid_5(5);
	can_bypass_5(6) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_6_q and addr_same_5(6) and addr_valid_5(6);
	can_bypass_5(7) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_7_q and addr_same_5(7) and addr_valid_5(7);
	can_bypass_5(8) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_8_q and addr_same_5(8) and addr_valid_5(8);
	can_bypass_5(9) <= ldq_alloc_5_q and not ldq_issue_5_q and stq_data_valid_9_q and addr_same_5(9) and addr_valid_5(9);
	can_bypass_6(0) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_0_q and addr_same_6(0) and addr_valid_6(0);
	can_bypass_6(1) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_1_q and addr_same_6(1) and addr_valid_6(1);
	can_bypass_6(2) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_2_q and addr_same_6(2) and addr_valid_6(2);
	can_bypass_6(3) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_3_q and addr_same_6(3) and addr_valid_6(3);
	can_bypass_6(4) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_4_q and addr_same_6(4) and addr_valid_6(4);
	can_bypass_6(5) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_5_q and addr_same_6(5) and addr_valid_6(5);
	can_bypass_6(6) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_6_q and addr_same_6(6) and addr_valid_6(6);
	can_bypass_6(7) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_7_q and addr_same_6(7) and addr_valid_6(7);
	can_bypass_6(8) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_8_q and addr_same_6(8) and addr_valid_6(8);
	can_bypass_6(9) <= ldq_alloc_6_q and not ldq_issue_6_q and stq_data_valid_9_q and addr_same_6(9) and addr_valid_6(9);
	can_bypass_7(0) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_0_q and addr_same_7(0) and addr_valid_7(0);
	can_bypass_7(1) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_1_q and addr_same_7(1) and addr_valid_7(1);
	can_bypass_7(2) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_2_q and addr_same_7(2) and addr_valid_7(2);
	can_bypass_7(3) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_3_q and addr_same_7(3) and addr_valid_7(3);
	can_bypass_7(4) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_4_q and addr_same_7(4) and addr_valid_7(4);
	can_bypass_7(5) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_5_q and addr_same_7(5) and addr_valid_7(5);
	can_bypass_7(6) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_6_q and addr_same_7(6) and addr_valid_7(6);
	can_bypass_7(7) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_7_q and addr_same_7(7) and addr_valid_7(7);
	can_bypass_7(8) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_8_q and addr_same_7(8) and addr_valid_7(8);
	can_bypass_7(9) <= ldq_alloc_7_q and not ldq_issue_7_q and stq_data_valid_9_q and addr_same_7(9) and addr_valid_7(9);
	can_bypass_8(0) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_0_q and addr_same_8(0) and addr_valid_8(0);
	can_bypass_8(1) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_1_q and addr_same_8(1) and addr_valid_8(1);
	can_bypass_8(2) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_2_q and addr_same_8(2) and addr_valid_8(2);
	can_bypass_8(3) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_3_q and addr_same_8(3) and addr_valid_8(3);
	can_bypass_8(4) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_4_q and addr_same_8(4) and addr_valid_8(4);
	can_bypass_8(5) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_5_q and addr_same_8(5) and addr_valid_8(5);
	can_bypass_8(6) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_6_q and addr_same_8(6) and addr_valid_8(6);
	can_bypass_8(7) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_7_q and addr_same_8(7) and addr_valid_8(7);
	can_bypass_8(8) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_8_q and addr_same_8(8) and addr_valid_8(8);
	can_bypass_8(9) <= ldq_alloc_8_q and not ldq_issue_8_q and stq_data_valid_9_q and addr_same_8(9) and addr_valid_8(9);
	can_bypass_9(0) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_0_q and addr_same_9(0) and addr_valid_9(0);
	can_bypass_9(1) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_1_q and addr_same_9(1) and addr_valid_9(1);
	can_bypass_9(2) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_2_q and addr_same_9(2) and addr_valid_9(2);
	can_bypass_9(3) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_3_q and addr_same_9(3) and addr_valid_9(3);
	can_bypass_9(4) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_4_q and addr_same_9(4) and addr_valid_9(4);
	can_bypass_9(5) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_5_q and addr_same_9(5) and addr_valid_9(5);
	can_bypass_9(6) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_6_q and addr_same_9(6) and addr_valid_9(6);
	can_bypass_9(7) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_7_q and addr_same_9(7) and addr_valid_9(7);
	can_bypass_9(8) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_8_q and addr_same_9(8) and addr_valid_9(8);
	can_bypass_9(9) <= ldq_alloc_9_q and not ldq_issue_9_q and stq_data_valid_9_q and addr_same_9(9) and addr_valid_9(9);
	-- Reduction Begin
	-- Reduce(load_conflict_0, ld_st_conflict_0, or)
	TEMP_23_res(0) <= ld_st_conflict_0(0) or ld_st_conflict_0(8);
	TEMP_23_res(1) <= ld_st_conflict_0(1) or ld_st_conflict_0(9);
	TEMP_23_res(2) <= ld_st_conflict_0(2);
	TEMP_23_res(3) <= ld_st_conflict_0(3);
	TEMP_23_res(4) <= ld_st_conflict_0(4);
	TEMP_23_res(5) <= ld_st_conflict_0(5);
	TEMP_23_res(6) <= ld_st_conflict_0(6);
	TEMP_23_res(7) <= ld_st_conflict_0(7);
	-- Layer End
	TEMP_24_res(0) <= TEMP_23_res(0) or TEMP_23_res(4);
	TEMP_24_res(1) <= TEMP_23_res(1) or TEMP_23_res(5);
	TEMP_24_res(2) <= TEMP_23_res(2) or TEMP_23_res(6);
	TEMP_24_res(3) <= TEMP_23_res(3) or TEMP_23_res(7);
	-- Layer End
	TEMP_25_res(0) <= TEMP_24_res(0) or TEMP_24_res(2);
	TEMP_25_res(1) <= TEMP_24_res(1) or TEMP_24_res(3);
	-- Layer End
	load_conflict_0 <= TEMP_25_res(0) or TEMP_25_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_1, ld_st_conflict_1, or)
	TEMP_26_res(0) <= ld_st_conflict_1(0) or ld_st_conflict_1(8);
	TEMP_26_res(1) <= ld_st_conflict_1(1) or ld_st_conflict_1(9);
	TEMP_26_res(2) <= ld_st_conflict_1(2);
	TEMP_26_res(3) <= ld_st_conflict_1(3);
	TEMP_26_res(4) <= ld_st_conflict_1(4);
	TEMP_26_res(5) <= ld_st_conflict_1(5);
	TEMP_26_res(6) <= ld_st_conflict_1(6);
	TEMP_26_res(7) <= ld_st_conflict_1(7);
	-- Layer End
	TEMP_27_res(0) <= TEMP_26_res(0) or TEMP_26_res(4);
	TEMP_27_res(1) <= TEMP_26_res(1) or TEMP_26_res(5);
	TEMP_27_res(2) <= TEMP_26_res(2) or TEMP_26_res(6);
	TEMP_27_res(3) <= TEMP_26_res(3) or TEMP_26_res(7);
	-- Layer End
	TEMP_28_res(0) <= TEMP_27_res(0) or TEMP_27_res(2);
	TEMP_28_res(1) <= TEMP_27_res(1) or TEMP_27_res(3);
	-- Layer End
	load_conflict_1 <= TEMP_28_res(0) or TEMP_28_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_2, ld_st_conflict_2, or)
	TEMP_29_res(0) <= ld_st_conflict_2(0) or ld_st_conflict_2(8);
	TEMP_29_res(1) <= ld_st_conflict_2(1) or ld_st_conflict_2(9);
	TEMP_29_res(2) <= ld_st_conflict_2(2);
	TEMP_29_res(3) <= ld_st_conflict_2(3);
	TEMP_29_res(4) <= ld_st_conflict_2(4);
	TEMP_29_res(5) <= ld_st_conflict_2(5);
	TEMP_29_res(6) <= ld_st_conflict_2(6);
	TEMP_29_res(7) <= ld_st_conflict_2(7);
	-- Layer End
	TEMP_30_res(0) <= TEMP_29_res(0) or TEMP_29_res(4);
	TEMP_30_res(1) <= TEMP_29_res(1) or TEMP_29_res(5);
	TEMP_30_res(2) <= TEMP_29_res(2) or TEMP_29_res(6);
	TEMP_30_res(3) <= TEMP_29_res(3) or TEMP_29_res(7);
	-- Layer End
	TEMP_31_res(0) <= TEMP_30_res(0) or TEMP_30_res(2);
	TEMP_31_res(1) <= TEMP_30_res(1) or TEMP_30_res(3);
	-- Layer End
	load_conflict_2 <= TEMP_31_res(0) or TEMP_31_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_3, ld_st_conflict_3, or)
	TEMP_32_res(0) <= ld_st_conflict_3(0) or ld_st_conflict_3(8);
	TEMP_32_res(1) <= ld_st_conflict_3(1) or ld_st_conflict_3(9);
	TEMP_32_res(2) <= ld_st_conflict_3(2);
	TEMP_32_res(3) <= ld_st_conflict_3(3);
	TEMP_32_res(4) <= ld_st_conflict_3(4);
	TEMP_32_res(5) <= ld_st_conflict_3(5);
	TEMP_32_res(6) <= ld_st_conflict_3(6);
	TEMP_32_res(7) <= ld_st_conflict_3(7);
	-- Layer End
	TEMP_33_res(0) <= TEMP_32_res(0) or TEMP_32_res(4);
	TEMP_33_res(1) <= TEMP_32_res(1) or TEMP_32_res(5);
	TEMP_33_res(2) <= TEMP_32_res(2) or TEMP_32_res(6);
	TEMP_33_res(3) <= TEMP_32_res(3) or TEMP_32_res(7);
	-- Layer End
	TEMP_34_res(0) <= TEMP_33_res(0) or TEMP_33_res(2);
	TEMP_34_res(1) <= TEMP_33_res(1) or TEMP_33_res(3);
	-- Layer End
	load_conflict_3 <= TEMP_34_res(0) or TEMP_34_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_4, ld_st_conflict_4, or)
	TEMP_35_res(0) <= ld_st_conflict_4(0) or ld_st_conflict_4(8);
	TEMP_35_res(1) <= ld_st_conflict_4(1) or ld_st_conflict_4(9);
	TEMP_35_res(2) <= ld_st_conflict_4(2);
	TEMP_35_res(3) <= ld_st_conflict_4(3);
	TEMP_35_res(4) <= ld_st_conflict_4(4);
	TEMP_35_res(5) <= ld_st_conflict_4(5);
	TEMP_35_res(6) <= ld_st_conflict_4(6);
	TEMP_35_res(7) <= ld_st_conflict_4(7);
	-- Layer End
	TEMP_36_res(0) <= TEMP_35_res(0) or TEMP_35_res(4);
	TEMP_36_res(1) <= TEMP_35_res(1) or TEMP_35_res(5);
	TEMP_36_res(2) <= TEMP_35_res(2) or TEMP_35_res(6);
	TEMP_36_res(3) <= TEMP_35_res(3) or TEMP_35_res(7);
	-- Layer End
	TEMP_37_res(0) <= TEMP_36_res(0) or TEMP_36_res(2);
	TEMP_37_res(1) <= TEMP_36_res(1) or TEMP_36_res(3);
	-- Layer End
	load_conflict_4 <= TEMP_37_res(0) or TEMP_37_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_5, ld_st_conflict_5, or)
	TEMP_38_res(0) <= ld_st_conflict_5(0) or ld_st_conflict_5(8);
	TEMP_38_res(1) <= ld_st_conflict_5(1) or ld_st_conflict_5(9);
	TEMP_38_res(2) <= ld_st_conflict_5(2);
	TEMP_38_res(3) <= ld_st_conflict_5(3);
	TEMP_38_res(4) <= ld_st_conflict_5(4);
	TEMP_38_res(5) <= ld_st_conflict_5(5);
	TEMP_38_res(6) <= ld_st_conflict_5(6);
	TEMP_38_res(7) <= ld_st_conflict_5(7);
	-- Layer End
	TEMP_39_res(0) <= TEMP_38_res(0) or TEMP_38_res(4);
	TEMP_39_res(1) <= TEMP_38_res(1) or TEMP_38_res(5);
	TEMP_39_res(2) <= TEMP_38_res(2) or TEMP_38_res(6);
	TEMP_39_res(3) <= TEMP_38_res(3) or TEMP_38_res(7);
	-- Layer End
	TEMP_40_res(0) <= TEMP_39_res(0) or TEMP_39_res(2);
	TEMP_40_res(1) <= TEMP_39_res(1) or TEMP_39_res(3);
	-- Layer End
	load_conflict_5 <= TEMP_40_res(0) or TEMP_40_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_6, ld_st_conflict_6, or)
	TEMP_41_res(0) <= ld_st_conflict_6(0) or ld_st_conflict_6(8);
	TEMP_41_res(1) <= ld_st_conflict_6(1) or ld_st_conflict_6(9);
	TEMP_41_res(2) <= ld_st_conflict_6(2);
	TEMP_41_res(3) <= ld_st_conflict_6(3);
	TEMP_41_res(4) <= ld_st_conflict_6(4);
	TEMP_41_res(5) <= ld_st_conflict_6(5);
	TEMP_41_res(6) <= ld_st_conflict_6(6);
	TEMP_41_res(7) <= ld_st_conflict_6(7);
	-- Layer End
	TEMP_42_res(0) <= TEMP_41_res(0) or TEMP_41_res(4);
	TEMP_42_res(1) <= TEMP_41_res(1) or TEMP_41_res(5);
	TEMP_42_res(2) <= TEMP_41_res(2) or TEMP_41_res(6);
	TEMP_42_res(3) <= TEMP_41_res(3) or TEMP_41_res(7);
	-- Layer End
	TEMP_43_res(0) <= TEMP_42_res(0) or TEMP_42_res(2);
	TEMP_43_res(1) <= TEMP_42_res(1) or TEMP_42_res(3);
	-- Layer End
	load_conflict_6 <= TEMP_43_res(0) or TEMP_43_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_7, ld_st_conflict_7, or)
	TEMP_44_res(0) <= ld_st_conflict_7(0) or ld_st_conflict_7(8);
	TEMP_44_res(1) <= ld_st_conflict_7(1) or ld_st_conflict_7(9);
	TEMP_44_res(2) <= ld_st_conflict_7(2);
	TEMP_44_res(3) <= ld_st_conflict_7(3);
	TEMP_44_res(4) <= ld_st_conflict_7(4);
	TEMP_44_res(5) <= ld_st_conflict_7(5);
	TEMP_44_res(6) <= ld_st_conflict_7(6);
	TEMP_44_res(7) <= ld_st_conflict_7(7);
	-- Layer End
	TEMP_45_res(0) <= TEMP_44_res(0) or TEMP_44_res(4);
	TEMP_45_res(1) <= TEMP_44_res(1) or TEMP_44_res(5);
	TEMP_45_res(2) <= TEMP_44_res(2) or TEMP_44_res(6);
	TEMP_45_res(3) <= TEMP_44_res(3) or TEMP_44_res(7);
	-- Layer End
	TEMP_46_res(0) <= TEMP_45_res(0) or TEMP_45_res(2);
	TEMP_46_res(1) <= TEMP_45_res(1) or TEMP_45_res(3);
	-- Layer End
	load_conflict_7 <= TEMP_46_res(0) or TEMP_46_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_8, ld_st_conflict_8, or)
	TEMP_47_res(0) <= ld_st_conflict_8(0) or ld_st_conflict_8(8);
	TEMP_47_res(1) <= ld_st_conflict_8(1) or ld_st_conflict_8(9);
	TEMP_47_res(2) <= ld_st_conflict_8(2);
	TEMP_47_res(3) <= ld_st_conflict_8(3);
	TEMP_47_res(4) <= ld_st_conflict_8(4);
	TEMP_47_res(5) <= ld_st_conflict_8(5);
	TEMP_47_res(6) <= ld_st_conflict_8(6);
	TEMP_47_res(7) <= ld_st_conflict_8(7);
	-- Layer End
	TEMP_48_res(0) <= TEMP_47_res(0) or TEMP_47_res(4);
	TEMP_48_res(1) <= TEMP_47_res(1) or TEMP_47_res(5);
	TEMP_48_res(2) <= TEMP_47_res(2) or TEMP_47_res(6);
	TEMP_48_res(3) <= TEMP_47_res(3) or TEMP_47_res(7);
	-- Layer End
	TEMP_49_res(0) <= TEMP_48_res(0) or TEMP_48_res(2);
	TEMP_49_res(1) <= TEMP_48_res(1) or TEMP_48_res(3);
	-- Layer End
	load_conflict_8 <= TEMP_49_res(0) or TEMP_49_res(1);
	-- Reduction End

	-- Reduction Begin
	-- Reduce(load_conflict_9, ld_st_conflict_9, or)
	TEMP_50_res(0) <= ld_st_conflict_9(0) or ld_st_conflict_9(8);
	TEMP_50_res(1) <= ld_st_conflict_9(1) or ld_st_conflict_9(9);
	TEMP_50_res(2) <= ld_st_conflict_9(2);
	TEMP_50_res(3) <= ld_st_conflict_9(3);
	TEMP_50_res(4) <= ld_st_conflict_9(4);
	TEMP_50_res(5) <= ld_st_conflict_9(5);
	TEMP_50_res(6) <= ld_st_conflict_9(6);
	TEMP_50_res(7) <= ld_st_conflict_9(7);
	-- Layer End
	TEMP_51_res(0) <= TEMP_50_res(0) or TEMP_50_res(4);
	TEMP_51_res(1) <= TEMP_50_res(1) or TEMP_50_res(5);
	TEMP_51_res(2) <= TEMP_50_res(2) or TEMP_50_res(6);
	TEMP_51_res(3) <= TEMP_50_res(3) or TEMP_50_res(7);
	-- Layer End
	TEMP_52_res(0) <= TEMP_51_res(0) or TEMP_51_res(2);
	TEMP_52_res(1) <= TEMP_51_res(1) or TEMP_51_res(3);
	-- Layer End
	load_conflict_9 <= TEMP_52_res(0) or TEMP_52_res(1);
	-- Reduction End

	load_req_valid_0 <= ldq_alloc_0_q and not ldq_issue_0_q and ldq_addr_valid_0_q;
	load_req_valid_1 <= ldq_alloc_1_q and not ldq_issue_1_q and ldq_addr_valid_1_q;
	load_req_valid_2 <= ldq_alloc_2_q and not ldq_issue_2_q and ldq_addr_valid_2_q;
	load_req_valid_3 <= ldq_alloc_3_q and not ldq_issue_3_q and ldq_addr_valid_3_q;
	load_req_valid_4 <= ldq_alloc_4_q and not ldq_issue_4_q and ldq_addr_valid_4_q;
	load_req_valid_5 <= ldq_alloc_5_q and not ldq_issue_5_q and ldq_addr_valid_5_q;
	load_req_valid_6 <= ldq_alloc_6_q and not ldq_issue_6_q and ldq_addr_valid_6_q;
	load_req_valid_7 <= ldq_alloc_7_q and not ldq_issue_7_q and ldq_addr_valid_7_q;
	load_req_valid_8 <= ldq_alloc_8_q and not ldq_issue_8_q and ldq_addr_valid_8_q;
	load_req_valid_9 <= ldq_alloc_9_q and not ldq_issue_9_q and ldq_addr_valid_9_q;
	can_load_0 <= not load_conflict_0 and load_req_valid_0;
	can_load_1 <= not load_conflict_1 and load_req_valid_1;
	can_load_2 <= not load_conflict_2 and load_req_valid_2;
	can_load_3 <= not load_conflict_3 and load_req_valid_3;
	can_load_4 <= not load_conflict_4 and load_req_valid_4;
	can_load_5 <= not load_conflict_5 and load_req_valid_5;
	can_load_6 <= not load_conflict_6 and load_req_valid_6;
	can_load_7 <= not load_conflict_7 and load_req_valid_7;
	can_load_8 <= not load_conflict_8 and load_req_valid_8;
	can_load_9 <= not load_conflict_9 and load_req_valid_9;
	-- Priority Masking Begin
	-- CyclicPriorityMask(load_idx_oh_0, can_load, ldq_head_oh)
	TEMP_53_double_in(0) <= can_load_0;
	TEMP_53_double_in(10) <= can_load_0;
	TEMP_53_double_in(1) <= can_load_1;
	TEMP_53_double_in(11) <= can_load_1;
	TEMP_53_double_in(2) <= can_load_2;
	TEMP_53_double_in(12) <= can_load_2;
	TEMP_53_double_in(3) <= can_load_3;
	TEMP_53_double_in(13) <= can_load_3;
	TEMP_53_double_in(4) <= can_load_4;
	TEMP_53_double_in(14) <= can_load_4;
	TEMP_53_double_in(5) <= can_load_5;
	TEMP_53_double_in(15) <= can_load_5;
	TEMP_53_double_in(6) <= can_load_6;
	TEMP_53_double_in(16) <= can_load_6;
	TEMP_53_double_in(7) <= can_load_7;
	TEMP_53_double_in(17) <= can_load_7;
	TEMP_53_double_in(8) <= can_load_8;
	TEMP_53_double_in(18) <= can_load_8;
	TEMP_53_double_in(9) <= can_load_9;
	TEMP_53_double_in(19) <= can_load_9;
	TEMP_53_double_out <= TEMP_53_double_in and not std_logic_vector( unsigned( TEMP_53_double_in ) - unsigned( "0000000000" & ldq_head_oh ) );
	load_idx_oh_0 <= TEMP_53_double_out(9 downto 0) or TEMP_53_double_out(19 downto 10);
	-- Priority Masking End

	-- Reduction Begin
	-- Reduce(load_en_0, can_load, or)
	TEMP_54_res_0 <= can_load_0 or can_load_8;
	TEMP_54_res_1 <= can_load_1 or can_load_9;
	TEMP_54_res_2 <= can_load_2;
	TEMP_54_res_3 <= can_load_3;
	TEMP_54_res_4 <= can_load_4;
	TEMP_54_res_5 <= can_load_5;
	TEMP_54_res_6 <= can_load_6;
	TEMP_54_res_7 <= can_load_7;
	-- Layer End
	TEMP_55_res_0 <= TEMP_54_res_0 or TEMP_54_res_4;
	TEMP_55_res_1 <= TEMP_54_res_1 or TEMP_54_res_5;
	TEMP_55_res_2 <= TEMP_54_res_2 or TEMP_54_res_6;
	TEMP_55_res_3 <= TEMP_54_res_3 or TEMP_54_res_7;
	-- Layer End
	TEMP_56_res_0 <= TEMP_55_res_0 or TEMP_55_res_2;
	TEMP_56_res_1 <= TEMP_55_res_1 or TEMP_55_res_3;
	-- Layer End
	load_en_0 <= TEMP_56_res_0 or TEMP_56_res_1;
	-- Reduction End

	st_ld_conflict(0) <= ldq_alloc_0_q and not store_is_older_0_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_0(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_0_q );
	st_ld_conflict(1) <= ldq_alloc_1_q and not store_is_older_1_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_1(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_1_q );
	st_ld_conflict(2) <= ldq_alloc_2_q and not store_is_older_2_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_2(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_2_q );
	st_ld_conflict(3) <= ldq_alloc_3_q and not store_is_older_3_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_3(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_3_q );
	st_ld_conflict(4) <= ldq_alloc_4_q and not store_is_older_4_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_4(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_4_q );
	st_ld_conflict(5) <= ldq_alloc_5_q and not store_is_older_5_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_5(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_5_q );
	st_ld_conflict(6) <= ldq_alloc_6_q and not store_is_older_6_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_6(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_6_q );
	st_ld_conflict(7) <= ldq_alloc_7_q and not store_is_older_7_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_7(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_7_q );
	st_ld_conflict(8) <= ldq_alloc_8_q and not store_is_older_8_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_8(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_8_q );
	st_ld_conflict(9) <= ldq_alloc_9_q and not store_is_older_9_q(to_integer(unsigned(stq_issue_q))) and ( addr_same_9(to_integer(unsigned(stq_issue_q))) or not ldq_addr_valid_9_q );
	-- Reduction Begin
	-- Reduce(store_conflict, st_ld_conflict, or)
	TEMP_57_res(0) <= st_ld_conflict(0) or st_ld_conflict(8);
	TEMP_57_res(1) <= st_ld_conflict(1) or st_ld_conflict(9);
	TEMP_57_res(2) <= st_ld_conflict(2);
	TEMP_57_res(3) <= st_ld_conflict(3);
	TEMP_57_res(4) <= st_ld_conflict(4);
	TEMP_57_res(5) <= st_ld_conflict(5);
	TEMP_57_res(6) <= st_ld_conflict(6);
	TEMP_57_res(7) <= st_ld_conflict(7);
	-- Layer End
	TEMP_58_res(0) <= TEMP_57_res(0) or TEMP_57_res(4);
	TEMP_58_res(1) <= TEMP_57_res(1) or TEMP_57_res(5);
	TEMP_58_res(2) <= TEMP_57_res(2) or TEMP_57_res(6);
	TEMP_58_res(3) <= TEMP_57_res(3) or TEMP_57_res(7);
	-- Layer End
	TEMP_59_res(0) <= TEMP_58_res(0) or TEMP_58_res(2);
	TEMP_59_res(1) <= TEMP_58_res(1) or TEMP_58_res(3);
	-- Layer End
	store_conflict <= TEMP_59_res(0) or TEMP_59_res(1);
	-- Reduction End

	-- MuxLookUp Begin
	-- MuxLookUp(store_valid, stq_alloc, stq_issue)
	store_valid <= 
	stq_alloc_0_q when (stq_issue_q = "0000") else
	stq_alloc_1_q when (stq_issue_q = "0001") else
	stq_alloc_2_q when (stq_issue_q = "0010") else
	stq_alloc_3_q when (stq_issue_q = "0011") else
	stq_alloc_4_q when (stq_issue_q = "0100") else
	stq_alloc_5_q when (stq_issue_q = "0101") else
	stq_alloc_6_q when (stq_issue_q = "0110") else
	stq_alloc_7_q when (stq_issue_q = "0111") else
	stq_alloc_8_q when (stq_issue_q = "1000") else
	stq_alloc_9_q when (stq_issue_q = "1001") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_data_valid, stq_data_valid, stq_issue)
	store_data_valid <= 
	stq_data_valid_0_q when (stq_issue_q = "0000") else
	stq_data_valid_1_q when (stq_issue_q = "0001") else
	stq_data_valid_2_q when (stq_issue_q = "0010") else
	stq_data_valid_3_q when (stq_issue_q = "0011") else
	stq_data_valid_4_q when (stq_issue_q = "0100") else
	stq_data_valid_5_q when (stq_issue_q = "0101") else
	stq_data_valid_6_q when (stq_issue_q = "0110") else
	stq_data_valid_7_q when (stq_issue_q = "0111") else
	stq_data_valid_8_q when (stq_issue_q = "1000") else
	stq_data_valid_9_q when (stq_issue_q = "1001") else
	'0';
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(store_addr_valid, stq_addr_valid, stq_issue)
	store_addr_valid <= 
	stq_addr_valid_0_q when (stq_issue_q = "0000") else
	stq_addr_valid_1_q when (stq_issue_q = "0001") else
	stq_addr_valid_2_q when (stq_issue_q = "0010") else
	stq_addr_valid_3_q when (stq_issue_q = "0011") else
	stq_addr_valid_4_q when (stq_issue_q = "0100") else
	stq_addr_valid_5_q when (stq_issue_q = "0101") else
	stq_addr_valid_6_q when (stq_issue_q = "0110") else
	stq_addr_valid_7_q when (stq_issue_q = "0111") else
	stq_addr_valid_8_q when (stq_issue_q = "1000") else
	stq_addr_valid_9_q when (stq_issue_q = "1001") else
	'0';
	-- MuxLookUp End

	store_en <= not store_conflict and store_valid and store_data_valid and store_addr_valid;
	store_idx <= stq_issue_q;
	-- Bits To One-Hot Begin
	-- BitsToOHSub1(stq_last_oh, stq_tail)
	stq_last_oh(0) <= '1' when stq_tail_q = "0001" else '0';
	stq_last_oh(1) <= '1' when stq_tail_q = "0010" else '0';
	stq_last_oh(2) <= '1' when stq_tail_q = "0011" else '0';
	stq_last_oh(3) <= '1' when stq_tail_q = "0100" else '0';
	stq_last_oh(4) <= '1' when stq_tail_q = "0101" else '0';
	stq_last_oh(5) <= '1' when stq_tail_q = "0110" else '0';
	stq_last_oh(6) <= '1' when stq_tail_q = "0111" else '0';
	stq_last_oh(7) <= '1' when stq_tail_q = "1000" else '0';
	stq_last_oh(8) <= '1' when stq_tail_q = "1001" else '0';
	stq_last_oh(9) <= '1' when stq_tail_q = "0000" else '0';
	-- Bits To One-Hot End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_0, ld_st_conflict_0, stq_last_oh)
	TEMP_60_double_in(0) <= ld_st_conflict_0(9);
	TEMP_60_double_in(10) <= ld_st_conflict_0(9);
	TEMP_60_double_in(1) <= ld_st_conflict_0(8);
	TEMP_60_double_in(11) <= ld_st_conflict_0(8);
	TEMP_60_double_in(2) <= ld_st_conflict_0(7);
	TEMP_60_double_in(12) <= ld_st_conflict_0(7);
	TEMP_60_double_in(3) <= ld_st_conflict_0(6);
	TEMP_60_double_in(13) <= ld_st_conflict_0(6);
	TEMP_60_double_in(4) <= ld_st_conflict_0(5);
	TEMP_60_double_in(14) <= ld_st_conflict_0(5);
	TEMP_60_double_in(5) <= ld_st_conflict_0(4);
	TEMP_60_double_in(15) <= ld_st_conflict_0(4);
	TEMP_60_double_in(6) <= ld_st_conflict_0(3);
	TEMP_60_double_in(16) <= ld_st_conflict_0(3);
	TEMP_60_double_in(7) <= ld_st_conflict_0(2);
	TEMP_60_double_in(17) <= ld_st_conflict_0(2);
	TEMP_60_double_in(8) <= ld_st_conflict_0(1);
	TEMP_60_double_in(18) <= ld_st_conflict_0(1);
	TEMP_60_double_in(9) <= ld_st_conflict_0(0);
	TEMP_60_double_in(19) <= ld_st_conflict_0(0);
	TEMP_60_base_rev(0) <= stq_last_oh(9);
	TEMP_60_base_rev(1) <= stq_last_oh(8);
	TEMP_60_base_rev(2) <= stq_last_oh(7);
	TEMP_60_base_rev(3) <= stq_last_oh(6);
	TEMP_60_base_rev(4) <= stq_last_oh(5);
	TEMP_60_base_rev(5) <= stq_last_oh(4);
	TEMP_60_base_rev(6) <= stq_last_oh(3);
	TEMP_60_base_rev(7) <= stq_last_oh(2);
	TEMP_60_base_rev(8) <= stq_last_oh(1);
	TEMP_60_base_rev(9) <= stq_last_oh(0);
	TEMP_60_double_out <= TEMP_60_double_in and not std_logic_vector( unsigned( TEMP_60_double_in ) - unsigned( "0000000000" & TEMP_60_base_rev ) );
	bypass_idx_oh_0(9) <= TEMP_60_double_out(0) or TEMP_60_double_out(10);
	bypass_idx_oh_0(8) <= TEMP_60_double_out(1) or TEMP_60_double_out(11);
	bypass_idx_oh_0(7) <= TEMP_60_double_out(2) or TEMP_60_double_out(12);
	bypass_idx_oh_0(6) <= TEMP_60_double_out(3) or TEMP_60_double_out(13);
	bypass_idx_oh_0(5) <= TEMP_60_double_out(4) or TEMP_60_double_out(14);
	bypass_idx_oh_0(4) <= TEMP_60_double_out(5) or TEMP_60_double_out(15);
	bypass_idx_oh_0(3) <= TEMP_60_double_out(6) or TEMP_60_double_out(16);
	bypass_idx_oh_0(2) <= TEMP_60_double_out(7) or TEMP_60_double_out(17);
	bypass_idx_oh_0(1) <= TEMP_60_double_out(8) or TEMP_60_double_out(18);
	bypass_idx_oh_0(0) <= TEMP_60_double_out(9) or TEMP_60_double_out(19);
	-- Priority Masking End

	bypass_en_vec_0 <= bypass_idx_oh_0 and can_bypass_0;
	-- Reduction Begin
	-- Reduce(bypass_en_0, bypass_en_vec_0, or)
	TEMP_61_res(0) <= bypass_en_vec_0(0) or bypass_en_vec_0(8);
	TEMP_61_res(1) <= bypass_en_vec_0(1) or bypass_en_vec_0(9);
	TEMP_61_res(2) <= bypass_en_vec_0(2);
	TEMP_61_res(3) <= bypass_en_vec_0(3);
	TEMP_61_res(4) <= bypass_en_vec_0(4);
	TEMP_61_res(5) <= bypass_en_vec_0(5);
	TEMP_61_res(6) <= bypass_en_vec_0(6);
	TEMP_61_res(7) <= bypass_en_vec_0(7);
	-- Layer End
	TEMP_62_res(0) <= TEMP_61_res(0) or TEMP_61_res(4);
	TEMP_62_res(1) <= TEMP_61_res(1) or TEMP_61_res(5);
	TEMP_62_res(2) <= TEMP_61_res(2) or TEMP_61_res(6);
	TEMP_62_res(3) <= TEMP_61_res(3) or TEMP_61_res(7);
	-- Layer End
	TEMP_63_res(0) <= TEMP_62_res(0) or TEMP_62_res(2);
	TEMP_63_res(1) <= TEMP_62_res(1) or TEMP_62_res(3);
	-- Layer End
	bypass_en_0 <= TEMP_63_res(0) or TEMP_63_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_1, ld_st_conflict_1, stq_last_oh)
	TEMP_64_double_in(0) <= ld_st_conflict_1(9);
	TEMP_64_double_in(10) <= ld_st_conflict_1(9);
	TEMP_64_double_in(1) <= ld_st_conflict_1(8);
	TEMP_64_double_in(11) <= ld_st_conflict_1(8);
	TEMP_64_double_in(2) <= ld_st_conflict_1(7);
	TEMP_64_double_in(12) <= ld_st_conflict_1(7);
	TEMP_64_double_in(3) <= ld_st_conflict_1(6);
	TEMP_64_double_in(13) <= ld_st_conflict_1(6);
	TEMP_64_double_in(4) <= ld_st_conflict_1(5);
	TEMP_64_double_in(14) <= ld_st_conflict_1(5);
	TEMP_64_double_in(5) <= ld_st_conflict_1(4);
	TEMP_64_double_in(15) <= ld_st_conflict_1(4);
	TEMP_64_double_in(6) <= ld_st_conflict_1(3);
	TEMP_64_double_in(16) <= ld_st_conflict_1(3);
	TEMP_64_double_in(7) <= ld_st_conflict_1(2);
	TEMP_64_double_in(17) <= ld_st_conflict_1(2);
	TEMP_64_double_in(8) <= ld_st_conflict_1(1);
	TEMP_64_double_in(18) <= ld_st_conflict_1(1);
	TEMP_64_double_in(9) <= ld_st_conflict_1(0);
	TEMP_64_double_in(19) <= ld_st_conflict_1(0);
	TEMP_64_base_rev(0) <= stq_last_oh(9);
	TEMP_64_base_rev(1) <= stq_last_oh(8);
	TEMP_64_base_rev(2) <= stq_last_oh(7);
	TEMP_64_base_rev(3) <= stq_last_oh(6);
	TEMP_64_base_rev(4) <= stq_last_oh(5);
	TEMP_64_base_rev(5) <= stq_last_oh(4);
	TEMP_64_base_rev(6) <= stq_last_oh(3);
	TEMP_64_base_rev(7) <= stq_last_oh(2);
	TEMP_64_base_rev(8) <= stq_last_oh(1);
	TEMP_64_base_rev(9) <= stq_last_oh(0);
	TEMP_64_double_out <= TEMP_64_double_in and not std_logic_vector( unsigned( TEMP_64_double_in ) - unsigned( "0000000000" & TEMP_64_base_rev ) );
	bypass_idx_oh_1(9) <= TEMP_64_double_out(0) or TEMP_64_double_out(10);
	bypass_idx_oh_1(8) <= TEMP_64_double_out(1) or TEMP_64_double_out(11);
	bypass_idx_oh_1(7) <= TEMP_64_double_out(2) or TEMP_64_double_out(12);
	bypass_idx_oh_1(6) <= TEMP_64_double_out(3) or TEMP_64_double_out(13);
	bypass_idx_oh_1(5) <= TEMP_64_double_out(4) or TEMP_64_double_out(14);
	bypass_idx_oh_1(4) <= TEMP_64_double_out(5) or TEMP_64_double_out(15);
	bypass_idx_oh_1(3) <= TEMP_64_double_out(6) or TEMP_64_double_out(16);
	bypass_idx_oh_1(2) <= TEMP_64_double_out(7) or TEMP_64_double_out(17);
	bypass_idx_oh_1(1) <= TEMP_64_double_out(8) or TEMP_64_double_out(18);
	bypass_idx_oh_1(0) <= TEMP_64_double_out(9) or TEMP_64_double_out(19);
	-- Priority Masking End

	bypass_en_vec_1 <= bypass_idx_oh_1 and can_bypass_1;
	-- Reduction Begin
	-- Reduce(bypass_en_1, bypass_en_vec_1, or)
	TEMP_65_res(0) <= bypass_en_vec_1(0) or bypass_en_vec_1(8);
	TEMP_65_res(1) <= bypass_en_vec_1(1) or bypass_en_vec_1(9);
	TEMP_65_res(2) <= bypass_en_vec_1(2);
	TEMP_65_res(3) <= bypass_en_vec_1(3);
	TEMP_65_res(4) <= bypass_en_vec_1(4);
	TEMP_65_res(5) <= bypass_en_vec_1(5);
	TEMP_65_res(6) <= bypass_en_vec_1(6);
	TEMP_65_res(7) <= bypass_en_vec_1(7);
	-- Layer End
	TEMP_66_res(0) <= TEMP_65_res(0) or TEMP_65_res(4);
	TEMP_66_res(1) <= TEMP_65_res(1) or TEMP_65_res(5);
	TEMP_66_res(2) <= TEMP_65_res(2) or TEMP_65_res(6);
	TEMP_66_res(3) <= TEMP_65_res(3) or TEMP_65_res(7);
	-- Layer End
	TEMP_67_res(0) <= TEMP_66_res(0) or TEMP_66_res(2);
	TEMP_67_res(1) <= TEMP_66_res(1) or TEMP_66_res(3);
	-- Layer End
	bypass_en_1 <= TEMP_67_res(0) or TEMP_67_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_2, ld_st_conflict_2, stq_last_oh)
	TEMP_68_double_in(0) <= ld_st_conflict_2(9);
	TEMP_68_double_in(10) <= ld_st_conflict_2(9);
	TEMP_68_double_in(1) <= ld_st_conflict_2(8);
	TEMP_68_double_in(11) <= ld_st_conflict_2(8);
	TEMP_68_double_in(2) <= ld_st_conflict_2(7);
	TEMP_68_double_in(12) <= ld_st_conflict_2(7);
	TEMP_68_double_in(3) <= ld_st_conflict_2(6);
	TEMP_68_double_in(13) <= ld_st_conflict_2(6);
	TEMP_68_double_in(4) <= ld_st_conflict_2(5);
	TEMP_68_double_in(14) <= ld_st_conflict_2(5);
	TEMP_68_double_in(5) <= ld_st_conflict_2(4);
	TEMP_68_double_in(15) <= ld_st_conflict_2(4);
	TEMP_68_double_in(6) <= ld_st_conflict_2(3);
	TEMP_68_double_in(16) <= ld_st_conflict_2(3);
	TEMP_68_double_in(7) <= ld_st_conflict_2(2);
	TEMP_68_double_in(17) <= ld_st_conflict_2(2);
	TEMP_68_double_in(8) <= ld_st_conflict_2(1);
	TEMP_68_double_in(18) <= ld_st_conflict_2(1);
	TEMP_68_double_in(9) <= ld_st_conflict_2(0);
	TEMP_68_double_in(19) <= ld_st_conflict_2(0);
	TEMP_68_base_rev(0) <= stq_last_oh(9);
	TEMP_68_base_rev(1) <= stq_last_oh(8);
	TEMP_68_base_rev(2) <= stq_last_oh(7);
	TEMP_68_base_rev(3) <= stq_last_oh(6);
	TEMP_68_base_rev(4) <= stq_last_oh(5);
	TEMP_68_base_rev(5) <= stq_last_oh(4);
	TEMP_68_base_rev(6) <= stq_last_oh(3);
	TEMP_68_base_rev(7) <= stq_last_oh(2);
	TEMP_68_base_rev(8) <= stq_last_oh(1);
	TEMP_68_base_rev(9) <= stq_last_oh(0);
	TEMP_68_double_out <= TEMP_68_double_in and not std_logic_vector( unsigned( TEMP_68_double_in ) - unsigned( "0000000000" & TEMP_68_base_rev ) );
	bypass_idx_oh_2(9) <= TEMP_68_double_out(0) or TEMP_68_double_out(10);
	bypass_idx_oh_2(8) <= TEMP_68_double_out(1) or TEMP_68_double_out(11);
	bypass_idx_oh_2(7) <= TEMP_68_double_out(2) or TEMP_68_double_out(12);
	bypass_idx_oh_2(6) <= TEMP_68_double_out(3) or TEMP_68_double_out(13);
	bypass_idx_oh_2(5) <= TEMP_68_double_out(4) or TEMP_68_double_out(14);
	bypass_idx_oh_2(4) <= TEMP_68_double_out(5) or TEMP_68_double_out(15);
	bypass_idx_oh_2(3) <= TEMP_68_double_out(6) or TEMP_68_double_out(16);
	bypass_idx_oh_2(2) <= TEMP_68_double_out(7) or TEMP_68_double_out(17);
	bypass_idx_oh_2(1) <= TEMP_68_double_out(8) or TEMP_68_double_out(18);
	bypass_idx_oh_2(0) <= TEMP_68_double_out(9) or TEMP_68_double_out(19);
	-- Priority Masking End

	bypass_en_vec_2 <= bypass_idx_oh_2 and can_bypass_2;
	-- Reduction Begin
	-- Reduce(bypass_en_2, bypass_en_vec_2, or)
	TEMP_69_res(0) <= bypass_en_vec_2(0) or bypass_en_vec_2(8);
	TEMP_69_res(1) <= bypass_en_vec_2(1) or bypass_en_vec_2(9);
	TEMP_69_res(2) <= bypass_en_vec_2(2);
	TEMP_69_res(3) <= bypass_en_vec_2(3);
	TEMP_69_res(4) <= bypass_en_vec_2(4);
	TEMP_69_res(5) <= bypass_en_vec_2(5);
	TEMP_69_res(6) <= bypass_en_vec_2(6);
	TEMP_69_res(7) <= bypass_en_vec_2(7);
	-- Layer End
	TEMP_70_res(0) <= TEMP_69_res(0) or TEMP_69_res(4);
	TEMP_70_res(1) <= TEMP_69_res(1) or TEMP_69_res(5);
	TEMP_70_res(2) <= TEMP_69_res(2) or TEMP_69_res(6);
	TEMP_70_res(3) <= TEMP_69_res(3) or TEMP_69_res(7);
	-- Layer End
	TEMP_71_res(0) <= TEMP_70_res(0) or TEMP_70_res(2);
	TEMP_71_res(1) <= TEMP_70_res(1) or TEMP_70_res(3);
	-- Layer End
	bypass_en_2 <= TEMP_71_res(0) or TEMP_71_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_3, ld_st_conflict_3, stq_last_oh)
	TEMP_72_double_in(0) <= ld_st_conflict_3(9);
	TEMP_72_double_in(10) <= ld_st_conflict_3(9);
	TEMP_72_double_in(1) <= ld_st_conflict_3(8);
	TEMP_72_double_in(11) <= ld_st_conflict_3(8);
	TEMP_72_double_in(2) <= ld_st_conflict_3(7);
	TEMP_72_double_in(12) <= ld_st_conflict_3(7);
	TEMP_72_double_in(3) <= ld_st_conflict_3(6);
	TEMP_72_double_in(13) <= ld_st_conflict_3(6);
	TEMP_72_double_in(4) <= ld_st_conflict_3(5);
	TEMP_72_double_in(14) <= ld_st_conflict_3(5);
	TEMP_72_double_in(5) <= ld_st_conflict_3(4);
	TEMP_72_double_in(15) <= ld_st_conflict_3(4);
	TEMP_72_double_in(6) <= ld_st_conflict_3(3);
	TEMP_72_double_in(16) <= ld_st_conflict_3(3);
	TEMP_72_double_in(7) <= ld_st_conflict_3(2);
	TEMP_72_double_in(17) <= ld_st_conflict_3(2);
	TEMP_72_double_in(8) <= ld_st_conflict_3(1);
	TEMP_72_double_in(18) <= ld_st_conflict_3(1);
	TEMP_72_double_in(9) <= ld_st_conflict_3(0);
	TEMP_72_double_in(19) <= ld_st_conflict_3(0);
	TEMP_72_base_rev(0) <= stq_last_oh(9);
	TEMP_72_base_rev(1) <= stq_last_oh(8);
	TEMP_72_base_rev(2) <= stq_last_oh(7);
	TEMP_72_base_rev(3) <= stq_last_oh(6);
	TEMP_72_base_rev(4) <= stq_last_oh(5);
	TEMP_72_base_rev(5) <= stq_last_oh(4);
	TEMP_72_base_rev(6) <= stq_last_oh(3);
	TEMP_72_base_rev(7) <= stq_last_oh(2);
	TEMP_72_base_rev(8) <= stq_last_oh(1);
	TEMP_72_base_rev(9) <= stq_last_oh(0);
	TEMP_72_double_out <= TEMP_72_double_in and not std_logic_vector( unsigned( TEMP_72_double_in ) - unsigned( "0000000000" & TEMP_72_base_rev ) );
	bypass_idx_oh_3(9) <= TEMP_72_double_out(0) or TEMP_72_double_out(10);
	bypass_idx_oh_3(8) <= TEMP_72_double_out(1) or TEMP_72_double_out(11);
	bypass_idx_oh_3(7) <= TEMP_72_double_out(2) or TEMP_72_double_out(12);
	bypass_idx_oh_3(6) <= TEMP_72_double_out(3) or TEMP_72_double_out(13);
	bypass_idx_oh_3(5) <= TEMP_72_double_out(4) or TEMP_72_double_out(14);
	bypass_idx_oh_3(4) <= TEMP_72_double_out(5) or TEMP_72_double_out(15);
	bypass_idx_oh_3(3) <= TEMP_72_double_out(6) or TEMP_72_double_out(16);
	bypass_idx_oh_3(2) <= TEMP_72_double_out(7) or TEMP_72_double_out(17);
	bypass_idx_oh_3(1) <= TEMP_72_double_out(8) or TEMP_72_double_out(18);
	bypass_idx_oh_3(0) <= TEMP_72_double_out(9) or TEMP_72_double_out(19);
	-- Priority Masking End

	bypass_en_vec_3 <= bypass_idx_oh_3 and can_bypass_3;
	-- Reduction Begin
	-- Reduce(bypass_en_3, bypass_en_vec_3, or)
	TEMP_73_res(0) <= bypass_en_vec_3(0) or bypass_en_vec_3(8);
	TEMP_73_res(1) <= bypass_en_vec_3(1) or bypass_en_vec_3(9);
	TEMP_73_res(2) <= bypass_en_vec_3(2);
	TEMP_73_res(3) <= bypass_en_vec_3(3);
	TEMP_73_res(4) <= bypass_en_vec_3(4);
	TEMP_73_res(5) <= bypass_en_vec_3(5);
	TEMP_73_res(6) <= bypass_en_vec_3(6);
	TEMP_73_res(7) <= bypass_en_vec_3(7);
	-- Layer End
	TEMP_74_res(0) <= TEMP_73_res(0) or TEMP_73_res(4);
	TEMP_74_res(1) <= TEMP_73_res(1) or TEMP_73_res(5);
	TEMP_74_res(2) <= TEMP_73_res(2) or TEMP_73_res(6);
	TEMP_74_res(3) <= TEMP_73_res(3) or TEMP_73_res(7);
	-- Layer End
	TEMP_75_res(0) <= TEMP_74_res(0) or TEMP_74_res(2);
	TEMP_75_res(1) <= TEMP_74_res(1) or TEMP_74_res(3);
	-- Layer End
	bypass_en_3 <= TEMP_75_res(0) or TEMP_75_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_4, ld_st_conflict_4, stq_last_oh)
	TEMP_76_double_in(0) <= ld_st_conflict_4(9);
	TEMP_76_double_in(10) <= ld_st_conflict_4(9);
	TEMP_76_double_in(1) <= ld_st_conflict_4(8);
	TEMP_76_double_in(11) <= ld_st_conflict_4(8);
	TEMP_76_double_in(2) <= ld_st_conflict_4(7);
	TEMP_76_double_in(12) <= ld_st_conflict_4(7);
	TEMP_76_double_in(3) <= ld_st_conflict_4(6);
	TEMP_76_double_in(13) <= ld_st_conflict_4(6);
	TEMP_76_double_in(4) <= ld_st_conflict_4(5);
	TEMP_76_double_in(14) <= ld_st_conflict_4(5);
	TEMP_76_double_in(5) <= ld_st_conflict_4(4);
	TEMP_76_double_in(15) <= ld_st_conflict_4(4);
	TEMP_76_double_in(6) <= ld_st_conflict_4(3);
	TEMP_76_double_in(16) <= ld_st_conflict_4(3);
	TEMP_76_double_in(7) <= ld_st_conflict_4(2);
	TEMP_76_double_in(17) <= ld_st_conflict_4(2);
	TEMP_76_double_in(8) <= ld_st_conflict_4(1);
	TEMP_76_double_in(18) <= ld_st_conflict_4(1);
	TEMP_76_double_in(9) <= ld_st_conflict_4(0);
	TEMP_76_double_in(19) <= ld_st_conflict_4(0);
	TEMP_76_base_rev(0) <= stq_last_oh(9);
	TEMP_76_base_rev(1) <= stq_last_oh(8);
	TEMP_76_base_rev(2) <= stq_last_oh(7);
	TEMP_76_base_rev(3) <= stq_last_oh(6);
	TEMP_76_base_rev(4) <= stq_last_oh(5);
	TEMP_76_base_rev(5) <= stq_last_oh(4);
	TEMP_76_base_rev(6) <= stq_last_oh(3);
	TEMP_76_base_rev(7) <= stq_last_oh(2);
	TEMP_76_base_rev(8) <= stq_last_oh(1);
	TEMP_76_base_rev(9) <= stq_last_oh(0);
	TEMP_76_double_out <= TEMP_76_double_in and not std_logic_vector( unsigned( TEMP_76_double_in ) - unsigned( "0000000000" & TEMP_76_base_rev ) );
	bypass_idx_oh_4(9) <= TEMP_76_double_out(0) or TEMP_76_double_out(10);
	bypass_idx_oh_4(8) <= TEMP_76_double_out(1) or TEMP_76_double_out(11);
	bypass_idx_oh_4(7) <= TEMP_76_double_out(2) or TEMP_76_double_out(12);
	bypass_idx_oh_4(6) <= TEMP_76_double_out(3) or TEMP_76_double_out(13);
	bypass_idx_oh_4(5) <= TEMP_76_double_out(4) or TEMP_76_double_out(14);
	bypass_idx_oh_4(4) <= TEMP_76_double_out(5) or TEMP_76_double_out(15);
	bypass_idx_oh_4(3) <= TEMP_76_double_out(6) or TEMP_76_double_out(16);
	bypass_idx_oh_4(2) <= TEMP_76_double_out(7) or TEMP_76_double_out(17);
	bypass_idx_oh_4(1) <= TEMP_76_double_out(8) or TEMP_76_double_out(18);
	bypass_idx_oh_4(0) <= TEMP_76_double_out(9) or TEMP_76_double_out(19);
	-- Priority Masking End

	bypass_en_vec_4 <= bypass_idx_oh_4 and can_bypass_4;
	-- Reduction Begin
	-- Reduce(bypass_en_4, bypass_en_vec_4, or)
	TEMP_77_res(0) <= bypass_en_vec_4(0) or bypass_en_vec_4(8);
	TEMP_77_res(1) <= bypass_en_vec_4(1) or bypass_en_vec_4(9);
	TEMP_77_res(2) <= bypass_en_vec_4(2);
	TEMP_77_res(3) <= bypass_en_vec_4(3);
	TEMP_77_res(4) <= bypass_en_vec_4(4);
	TEMP_77_res(5) <= bypass_en_vec_4(5);
	TEMP_77_res(6) <= bypass_en_vec_4(6);
	TEMP_77_res(7) <= bypass_en_vec_4(7);
	-- Layer End
	TEMP_78_res(0) <= TEMP_77_res(0) or TEMP_77_res(4);
	TEMP_78_res(1) <= TEMP_77_res(1) or TEMP_77_res(5);
	TEMP_78_res(2) <= TEMP_77_res(2) or TEMP_77_res(6);
	TEMP_78_res(3) <= TEMP_77_res(3) or TEMP_77_res(7);
	-- Layer End
	TEMP_79_res(0) <= TEMP_78_res(0) or TEMP_78_res(2);
	TEMP_79_res(1) <= TEMP_78_res(1) or TEMP_78_res(3);
	-- Layer End
	bypass_en_4 <= TEMP_79_res(0) or TEMP_79_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_5, ld_st_conflict_5, stq_last_oh)
	TEMP_80_double_in(0) <= ld_st_conflict_5(9);
	TEMP_80_double_in(10) <= ld_st_conflict_5(9);
	TEMP_80_double_in(1) <= ld_st_conflict_5(8);
	TEMP_80_double_in(11) <= ld_st_conflict_5(8);
	TEMP_80_double_in(2) <= ld_st_conflict_5(7);
	TEMP_80_double_in(12) <= ld_st_conflict_5(7);
	TEMP_80_double_in(3) <= ld_st_conflict_5(6);
	TEMP_80_double_in(13) <= ld_st_conflict_5(6);
	TEMP_80_double_in(4) <= ld_st_conflict_5(5);
	TEMP_80_double_in(14) <= ld_st_conflict_5(5);
	TEMP_80_double_in(5) <= ld_st_conflict_5(4);
	TEMP_80_double_in(15) <= ld_st_conflict_5(4);
	TEMP_80_double_in(6) <= ld_st_conflict_5(3);
	TEMP_80_double_in(16) <= ld_st_conflict_5(3);
	TEMP_80_double_in(7) <= ld_st_conflict_5(2);
	TEMP_80_double_in(17) <= ld_st_conflict_5(2);
	TEMP_80_double_in(8) <= ld_st_conflict_5(1);
	TEMP_80_double_in(18) <= ld_st_conflict_5(1);
	TEMP_80_double_in(9) <= ld_st_conflict_5(0);
	TEMP_80_double_in(19) <= ld_st_conflict_5(0);
	TEMP_80_base_rev(0) <= stq_last_oh(9);
	TEMP_80_base_rev(1) <= stq_last_oh(8);
	TEMP_80_base_rev(2) <= stq_last_oh(7);
	TEMP_80_base_rev(3) <= stq_last_oh(6);
	TEMP_80_base_rev(4) <= stq_last_oh(5);
	TEMP_80_base_rev(5) <= stq_last_oh(4);
	TEMP_80_base_rev(6) <= stq_last_oh(3);
	TEMP_80_base_rev(7) <= stq_last_oh(2);
	TEMP_80_base_rev(8) <= stq_last_oh(1);
	TEMP_80_base_rev(9) <= stq_last_oh(0);
	TEMP_80_double_out <= TEMP_80_double_in and not std_logic_vector( unsigned( TEMP_80_double_in ) - unsigned( "0000000000" & TEMP_80_base_rev ) );
	bypass_idx_oh_5(9) <= TEMP_80_double_out(0) or TEMP_80_double_out(10);
	bypass_idx_oh_5(8) <= TEMP_80_double_out(1) or TEMP_80_double_out(11);
	bypass_idx_oh_5(7) <= TEMP_80_double_out(2) or TEMP_80_double_out(12);
	bypass_idx_oh_5(6) <= TEMP_80_double_out(3) or TEMP_80_double_out(13);
	bypass_idx_oh_5(5) <= TEMP_80_double_out(4) or TEMP_80_double_out(14);
	bypass_idx_oh_5(4) <= TEMP_80_double_out(5) or TEMP_80_double_out(15);
	bypass_idx_oh_5(3) <= TEMP_80_double_out(6) or TEMP_80_double_out(16);
	bypass_idx_oh_5(2) <= TEMP_80_double_out(7) or TEMP_80_double_out(17);
	bypass_idx_oh_5(1) <= TEMP_80_double_out(8) or TEMP_80_double_out(18);
	bypass_idx_oh_5(0) <= TEMP_80_double_out(9) or TEMP_80_double_out(19);
	-- Priority Masking End

	bypass_en_vec_5 <= bypass_idx_oh_5 and can_bypass_5;
	-- Reduction Begin
	-- Reduce(bypass_en_5, bypass_en_vec_5, or)
	TEMP_81_res(0) <= bypass_en_vec_5(0) or bypass_en_vec_5(8);
	TEMP_81_res(1) <= bypass_en_vec_5(1) or bypass_en_vec_5(9);
	TEMP_81_res(2) <= bypass_en_vec_5(2);
	TEMP_81_res(3) <= bypass_en_vec_5(3);
	TEMP_81_res(4) <= bypass_en_vec_5(4);
	TEMP_81_res(5) <= bypass_en_vec_5(5);
	TEMP_81_res(6) <= bypass_en_vec_5(6);
	TEMP_81_res(7) <= bypass_en_vec_5(7);
	-- Layer End
	TEMP_82_res(0) <= TEMP_81_res(0) or TEMP_81_res(4);
	TEMP_82_res(1) <= TEMP_81_res(1) or TEMP_81_res(5);
	TEMP_82_res(2) <= TEMP_81_res(2) or TEMP_81_res(6);
	TEMP_82_res(3) <= TEMP_81_res(3) or TEMP_81_res(7);
	-- Layer End
	TEMP_83_res(0) <= TEMP_82_res(0) or TEMP_82_res(2);
	TEMP_83_res(1) <= TEMP_82_res(1) or TEMP_82_res(3);
	-- Layer End
	bypass_en_5 <= TEMP_83_res(0) or TEMP_83_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_6, ld_st_conflict_6, stq_last_oh)
	TEMP_84_double_in(0) <= ld_st_conflict_6(9);
	TEMP_84_double_in(10) <= ld_st_conflict_6(9);
	TEMP_84_double_in(1) <= ld_st_conflict_6(8);
	TEMP_84_double_in(11) <= ld_st_conflict_6(8);
	TEMP_84_double_in(2) <= ld_st_conflict_6(7);
	TEMP_84_double_in(12) <= ld_st_conflict_6(7);
	TEMP_84_double_in(3) <= ld_st_conflict_6(6);
	TEMP_84_double_in(13) <= ld_st_conflict_6(6);
	TEMP_84_double_in(4) <= ld_st_conflict_6(5);
	TEMP_84_double_in(14) <= ld_st_conflict_6(5);
	TEMP_84_double_in(5) <= ld_st_conflict_6(4);
	TEMP_84_double_in(15) <= ld_st_conflict_6(4);
	TEMP_84_double_in(6) <= ld_st_conflict_6(3);
	TEMP_84_double_in(16) <= ld_st_conflict_6(3);
	TEMP_84_double_in(7) <= ld_st_conflict_6(2);
	TEMP_84_double_in(17) <= ld_st_conflict_6(2);
	TEMP_84_double_in(8) <= ld_st_conflict_6(1);
	TEMP_84_double_in(18) <= ld_st_conflict_6(1);
	TEMP_84_double_in(9) <= ld_st_conflict_6(0);
	TEMP_84_double_in(19) <= ld_st_conflict_6(0);
	TEMP_84_base_rev(0) <= stq_last_oh(9);
	TEMP_84_base_rev(1) <= stq_last_oh(8);
	TEMP_84_base_rev(2) <= stq_last_oh(7);
	TEMP_84_base_rev(3) <= stq_last_oh(6);
	TEMP_84_base_rev(4) <= stq_last_oh(5);
	TEMP_84_base_rev(5) <= stq_last_oh(4);
	TEMP_84_base_rev(6) <= stq_last_oh(3);
	TEMP_84_base_rev(7) <= stq_last_oh(2);
	TEMP_84_base_rev(8) <= stq_last_oh(1);
	TEMP_84_base_rev(9) <= stq_last_oh(0);
	TEMP_84_double_out <= TEMP_84_double_in and not std_logic_vector( unsigned( TEMP_84_double_in ) - unsigned( "0000000000" & TEMP_84_base_rev ) );
	bypass_idx_oh_6(9) <= TEMP_84_double_out(0) or TEMP_84_double_out(10);
	bypass_idx_oh_6(8) <= TEMP_84_double_out(1) or TEMP_84_double_out(11);
	bypass_idx_oh_6(7) <= TEMP_84_double_out(2) or TEMP_84_double_out(12);
	bypass_idx_oh_6(6) <= TEMP_84_double_out(3) or TEMP_84_double_out(13);
	bypass_idx_oh_6(5) <= TEMP_84_double_out(4) or TEMP_84_double_out(14);
	bypass_idx_oh_6(4) <= TEMP_84_double_out(5) or TEMP_84_double_out(15);
	bypass_idx_oh_6(3) <= TEMP_84_double_out(6) or TEMP_84_double_out(16);
	bypass_idx_oh_6(2) <= TEMP_84_double_out(7) or TEMP_84_double_out(17);
	bypass_idx_oh_6(1) <= TEMP_84_double_out(8) or TEMP_84_double_out(18);
	bypass_idx_oh_6(0) <= TEMP_84_double_out(9) or TEMP_84_double_out(19);
	-- Priority Masking End

	bypass_en_vec_6 <= bypass_idx_oh_6 and can_bypass_6;
	-- Reduction Begin
	-- Reduce(bypass_en_6, bypass_en_vec_6, or)
	TEMP_85_res(0) <= bypass_en_vec_6(0) or bypass_en_vec_6(8);
	TEMP_85_res(1) <= bypass_en_vec_6(1) or bypass_en_vec_6(9);
	TEMP_85_res(2) <= bypass_en_vec_6(2);
	TEMP_85_res(3) <= bypass_en_vec_6(3);
	TEMP_85_res(4) <= bypass_en_vec_6(4);
	TEMP_85_res(5) <= bypass_en_vec_6(5);
	TEMP_85_res(6) <= bypass_en_vec_6(6);
	TEMP_85_res(7) <= bypass_en_vec_6(7);
	-- Layer End
	TEMP_86_res(0) <= TEMP_85_res(0) or TEMP_85_res(4);
	TEMP_86_res(1) <= TEMP_85_res(1) or TEMP_85_res(5);
	TEMP_86_res(2) <= TEMP_85_res(2) or TEMP_85_res(6);
	TEMP_86_res(3) <= TEMP_85_res(3) or TEMP_85_res(7);
	-- Layer End
	TEMP_87_res(0) <= TEMP_86_res(0) or TEMP_86_res(2);
	TEMP_87_res(1) <= TEMP_86_res(1) or TEMP_86_res(3);
	-- Layer End
	bypass_en_6 <= TEMP_87_res(0) or TEMP_87_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_7, ld_st_conflict_7, stq_last_oh)
	TEMP_88_double_in(0) <= ld_st_conflict_7(9);
	TEMP_88_double_in(10) <= ld_st_conflict_7(9);
	TEMP_88_double_in(1) <= ld_st_conflict_7(8);
	TEMP_88_double_in(11) <= ld_st_conflict_7(8);
	TEMP_88_double_in(2) <= ld_st_conflict_7(7);
	TEMP_88_double_in(12) <= ld_st_conflict_7(7);
	TEMP_88_double_in(3) <= ld_st_conflict_7(6);
	TEMP_88_double_in(13) <= ld_st_conflict_7(6);
	TEMP_88_double_in(4) <= ld_st_conflict_7(5);
	TEMP_88_double_in(14) <= ld_st_conflict_7(5);
	TEMP_88_double_in(5) <= ld_st_conflict_7(4);
	TEMP_88_double_in(15) <= ld_st_conflict_7(4);
	TEMP_88_double_in(6) <= ld_st_conflict_7(3);
	TEMP_88_double_in(16) <= ld_st_conflict_7(3);
	TEMP_88_double_in(7) <= ld_st_conflict_7(2);
	TEMP_88_double_in(17) <= ld_st_conflict_7(2);
	TEMP_88_double_in(8) <= ld_st_conflict_7(1);
	TEMP_88_double_in(18) <= ld_st_conflict_7(1);
	TEMP_88_double_in(9) <= ld_st_conflict_7(0);
	TEMP_88_double_in(19) <= ld_st_conflict_7(0);
	TEMP_88_base_rev(0) <= stq_last_oh(9);
	TEMP_88_base_rev(1) <= stq_last_oh(8);
	TEMP_88_base_rev(2) <= stq_last_oh(7);
	TEMP_88_base_rev(3) <= stq_last_oh(6);
	TEMP_88_base_rev(4) <= stq_last_oh(5);
	TEMP_88_base_rev(5) <= stq_last_oh(4);
	TEMP_88_base_rev(6) <= stq_last_oh(3);
	TEMP_88_base_rev(7) <= stq_last_oh(2);
	TEMP_88_base_rev(8) <= stq_last_oh(1);
	TEMP_88_base_rev(9) <= stq_last_oh(0);
	TEMP_88_double_out <= TEMP_88_double_in and not std_logic_vector( unsigned( TEMP_88_double_in ) - unsigned( "0000000000" & TEMP_88_base_rev ) );
	bypass_idx_oh_7(9) <= TEMP_88_double_out(0) or TEMP_88_double_out(10);
	bypass_idx_oh_7(8) <= TEMP_88_double_out(1) or TEMP_88_double_out(11);
	bypass_idx_oh_7(7) <= TEMP_88_double_out(2) or TEMP_88_double_out(12);
	bypass_idx_oh_7(6) <= TEMP_88_double_out(3) or TEMP_88_double_out(13);
	bypass_idx_oh_7(5) <= TEMP_88_double_out(4) or TEMP_88_double_out(14);
	bypass_idx_oh_7(4) <= TEMP_88_double_out(5) or TEMP_88_double_out(15);
	bypass_idx_oh_7(3) <= TEMP_88_double_out(6) or TEMP_88_double_out(16);
	bypass_idx_oh_7(2) <= TEMP_88_double_out(7) or TEMP_88_double_out(17);
	bypass_idx_oh_7(1) <= TEMP_88_double_out(8) or TEMP_88_double_out(18);
	bypass_idx_oh_7(0) <= TEMP_88_double_out(9) or TEMP_88_double_out(19);
	-- Priority Masking End

	bypass_en_vec_7 <= bypass_idx_oh_7 and can_bypass_7;
	-- Reduction Begin
	-- Reduce(bypass_en_7, bypass_en_vec_7, or)
	TEMP_89_res(0) <= bypass_en_vec_7(0) or bypass_en_vec_7(8);
	TEMP_89_res(1) <= bypass_en_vec_7(1) or bypass_en_vec_7(9);
	TEMP_89_res(2) <= bypass_en_vec_7(2);
	TEMP_89_res(3) <= bypass_en_vec_7(3);
	TEMP_89_res(4) <= bypass_en_vec_7(4);
	TEMP_89_res(5) <= bypass_en_vec_7(5);
	TEMP_89_res(6) <= bypass_en_vec_7(6);
	TEMP_89_res(7) <= bypass_en_vec_7(7);
	-- Layer End
	TEMP_90_res(0) <= TEMP_89_res(0) or TEMP_89_res(4);
	TEMP_90_res(1) <= TEMP_89_res(1) or TEMP_89_res(5);
	TEMP_90_res(2) <= TEMP_89_res(2) or TEMP_89_res(6);
	TEMP_90_res(3) <= TEMP_89_res(3) or TEMP_89_res(7);
	-- Layer End
	TEMP_91_res(0) <= TEMP_90_res(0) or TEMP_90_res(2);
	TEMP_91_res(1) <= TEMP_90_res(1) or TEMP_90_res(3);
	-- Layer End
	bypass_en_7 <= TEMP_91_res(0) or TEMP_91_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_8, ld_st_conflict_8, stq_last_oh)
	TEMP_92_double_in(0) <= ld_st_conflict_8(9);
	TEMP_92_double_in(10) <= ld_st_conflict_8(9);
	TEMP_92_double_in(1) <= ld_st_conflict_8(8);
	TEMP_92_double_in(11) <= ld_st_conflict_8(8);
	TEMP_92_double_in(2) <= ld_st_conflict_8(7);
	TEMP_92_double_in(12) <= ld_st_conflict_8(7);
	TEMP_92_double_in(3) <= ld_st_conflict_8(6);
	TEMP_92_double_in(13) <= ld_st_conflict_8(6);
	TEMP_92_double_in(4) <= ld_st_conflict_8(5);
	TEMP_92_double_in(14) <= ld_st_conflict_8(5);
	TEMP_92_double_in(5) <= ld_st_conflict_8(4);
	TEMP_92_double_in(15) <= ld_st_conflict_8(4);
	TEMP_92_double_in(6) <= ld_st_conflict_8(3);
	TEMP_92_double_in(16) <= ld_st_conflict_8(3);
	TEMP_92_double_in(7) <= ld_st_conflict_8(2);
	TEMP_92_double_in(17) <= ld_st_conflict_8(2);
	TEMP_92_double_in(8) <= ld_st_conflict_8(1);
	TEMP_92_double_in(18) <= ld_st_conflict_8(1);
	TEMP_92_double_in(9) <= ld_st_conflict_8(0);
	TEMP_92_double_in(19) <= ld_st_conflict_8(0);
	TEMP_92_base_rev(0) <= stq_last_oh(9);
	TEMP_92_base_rev(1) <= stq_last_oh(8);
	TEMP_92_base_rev(2) <= stq_last_oh(7);
	TEMP_92_base_rev(3) <= stq_last_oh(6);
	TEMP_92_base_rev(4) <= stq_last_oh(5);
	TEMP_92_base_rev(5) <= stq_last_oh(4);
	TEMP_92_base_rev(6) <= stq_last_oh(3);
	TEMP_92_base_rev(7) <= stq_last_oh(2);
	TEMP_92_base_rev(8) <= stq_last_oh(1);
	TEMP_92_base_rev(9) <= stq_last_oh(0);
	TEMP_92_double_out <= TEMP_92_double_in and not std_logic_vector( unsigned( TEMP_92_double_in ) - unsigned( "0000000000" & TEMP_92_base_rev ) );
	bypass_idx_oh_8(9) <= TEMP_92_double_out(0) or TEMP_92_double_out(10);
	bypass_idx_oh_8(8) <= TEMP_92_double_out(1) or TEMP_92_double_out(11);
	bypass_idx_oh_8(7) <= TEMP_92_double_out(2) or TEMP_92_double_out(12);
	bypass_idx_oh_8(6) <= TEMP_92_double_out(3) or TEMP_92_double_out(13);
	bypass_idx_oh_8(5) <= TEMP_92_double_out(4) or TEMP_92_double_out(14);
	bypass_idx_oh_8(4) <= TEMP_92_double_out(5) or TEMP_92_double_out(15);
	bypass_idx_oh_8(3) <= TEMP_92_double_out(6) or TEMP_92_double_out(16);
	bypass_idx_oh_8(2) <= TEMP_92_double_out(7) or TEMP_92_double_out(17);
	bypass_idx_oh_8(1) <= TEMP_92_double_out(8) or TEMP_92_double_out(18);
	bypass_idx_oh_8(0) <= TEMP_92_double_out(9) or TEMP_92_double_out(19);
	-- Priority Masking End

	bypass_en_vec_8 <= bypass_idx_oh_8 and can_bypass_8;
	-- Reduction Begin
	-- Reduce(bypass_en_8, bypass_en_vec_8, or)
	TEMP_93_res(0) <= bypass_en_vec_8(0) or bypass_en_vec_8(8);
	TEMP_93_res(1) <= bypass_en_vec_8(1) or bypass_en_vec_8(9);
	TEMP_93_res(2) <= bypass_en_vec_8(2);
	TEMP_93_res(3) <= bypass_en_vec_8(3);
	TEMP_93_res(4) <= bypass_en_vec_8(4);
	TEMP_93_res(5) <= bypass_en_vec_8(5);
	TEMP_93_res(6) <= bypass_en_vec_8(6);
	TEMP_93_res(7) <= bypass_en_vec_8(7);
	-- Layer End
	TEMP_94_res(0) <= TEMP_93_res(0) or TEMP_93_res(4);
	TEMP_94_res(1) <= TEMP_93_res(1) or TEMP_93_res(5);
	TEMP_94_res(2) <= TEMP_93_res(2) or TEMP_93_res(6);
	TEMP_94_res(3) <= TEMP_93_res(3) or TEMP_93_res(7);
	-- Layer End
	TEMP_95_res(0) <= TEMP_94_res(0) or TEMP_94_res(2);
	TEMP_95_res(1) <= TEMP_94_res(1) or TEMP_94_res(3);
	-- Layer End
	bypass_en_8 <= TEMP_95_res(0) or TEMP_95_res(1);
	-- Reduction End

	-- Priority Masking Begin
	-- CyclicPriorityMask(bypass_idx_oh_9, ld_st_conflict_9, stq_last_oh)
	TEMP_96_double_in(0) <= ld_st_conflict_9(9);
	TEMP_96_double_in(10) <= ld_st_conflict_9(9);
	TEMP_96_double_in(1) <= ld_st_conflict_9(8);
	TEMP_96_double_in(11) <= ld_st_conflict_9(8);
	TEMP_96_double_in(2) <= ld_st_conflict_9(7);
	TEMP_96_double_in(12) <= ld_st_conflict_9(7);
	TEMP_96_double_in(3) <= ld_st_conflict_9(6);
	TEMP_96_double_in(13) <= ld_st_conflict_9(6);
	TEMP_96_double_in(4) <= ld_st_conflict_9(5);
	TEMP_96_double_in(14) <= ld_st_conflict_9(5);
	TEMP_96_double_in(5) <= ld_st_conflict_9(4);
	TEMP_96_double_in(15) <= ld_st_conflict_9(4);
	TEMP_96_double_in(6) <= ld_st_conflict_9(3);
	TEMP_96_double_in(16) <= ld_st_conflict_9(3);
	TEMP_96_double_in(7) <= ld_st_conflict_9(2);
	TEMP_96_double_in(17) <= ld_st_conflict_9(2);
	TEMP_96_double_in(8) <= ld_st_conflict_9(1);
	TEMP_96_double_in(18) <= ld_st_conflict_9(1);
	TEMP_96_double_in(9) <= ld_st_conflict_9(0);
	TEMP_96_double_in(19) <= ld_st_conflict_9(0);
	TEMP_96_base_rev(0) <= stq_last_oh(9);
	TEMP_96_base_rev(1) <= stq_last_oh(8);
	TEMP_96_base_rev(2) <= stq_last_oh(7);
	TEMP_96_base_rev(3) <= stq_last_oh(6);
	TEMP_96_base_rev(4) <= stq_last_oh(5);
	TEMP_96_base_rev(5) <= stq_last_oh(4);
	TEMP_96_base_rev(6) <= stq_last_oh(3);
	TEMP_96_base_rev(7) <= stq_last_oh(2);
	TEMP_96_base_rev(8) <= stq_last_oh(1);
	TEMP_96_base_rev(9) <= stq_last_oh(0);
	TEMP_96_double_out <= TEMP_96_double_in and not std_logic_vector( unsigned( TEMP_96_double_in ) - unsigned( "0000000000" & TEMP_96_base_rev ) );
	bypass_idx_oh_9(9) <= TEMP_96_double_out(0) or TEMP_96_double_out(10);
	bypass_idx_oh_9(8) <= TEMP_96_double_out(1) or TEMP_96_double_out(11);
	bypass_idx_oh_9(7) <= TEMP_96_double_out(2) or TEMP_96_double_out(12);
	bypass_idx_oh_9(6) <= TEMP_96_double_out(3) or TEMP_96_double_out(13);
	bypass_idx_oh_9(5) <= TEMP_96_double_out(4) or TEMP_96_double_out(14);
	bypass_idx_oh_9(4) <= TEMP_96_double_out(5) or TEMP_96_double_out(15);
	bypass_idx_oh_9(3) <= TEMP_96_double_out(6) or TEMP_96_double_out(16);
	bypass_idx_oh_9(2) <= TEMP_96_double_out(7) or TEMP_96_double_out(17);
	bypass_idx_oh_9(1) <= TEMP_96_double_out(8) or TEMP_96_double_out(18);
	bypass_idx_oh_9(0) <= TEMP_96_double_out(9) or TEMP_96_double_out(19);
	-- Priority Masking End

	bypass_en_vec_9 <= bypass_idx_oh_9 and can_bypass_9;
	-- Reduction Begin
	-- Reduce(bypass_en_9, bypass_en_vec_9, or)
	TEMP_97_res(0) <= bypass_en_vec_9(0) or bypass_en_vec_9(8);
	TEMP_97_res(1) <= bypass_en_vec_9(1) or bypass_en_vec_9(9);
	TEMP_97_res(2) <= bypass_en_vec_9(2);
	TEMP_97_res(3) <= bypass_en_vec_9(3);
	TEMP_97_res(4) <= bypass_en_vec_9(4);
	TEMP_97_res(5) <= bypass_en_vec_9(5);
	TEMP_97_res(6) <= bypass_en_vec_9(6);
	TEMP_97_res(7) <= bypass_en_vec_9(7);
	-- Layer End
	TEMP_98_res(0) <= TEMP_97_res(0) or TEMP_97_res(4);
	TEMP_98_res(1) <= TEMP_97_res(1) or TEMP_97_res(5);
	TEMP_98_res(2) <= TEMP_97_res(2) or TEMP_97_res(6);
	TEMP_98_res(3) <= TEMP_97_res(3) or TEMP_97_res(7);
	-- Layer End
	TEMP_99_res(0) <= TEMP_98_res(0) or TEMP_98_res(2);
	TEMP_99_res(1) <= TEMP_98_res(1) or TEMP_98_res(3);
	-- Layer End
	bypass_en_9 <= TEMP_99_res(0) or TEMP_99_res(1);
	-- Reduction End

	rreq_valid_0_o <= load_en_0;
	-- One-Hot To Bits Begin
	-- OHToBits(rreq_id_0, load_idx_oh_0)
	TEMP_100_in_0_0 <= '0';
	TEMP_100_in_0_1 <= load_idx_oh_0(1);
	TEMP_100_in_0_2 <= '0';
	TEMP_100_in_0_3 <= load_idx_oh_0(3);
	TEMP_100_in_0_4 <= '0';
	TEMP_100_in_0_5 <= load_idx_oh_0(5);
	TEMP_100_in_0_6 <= '0';
	TEMP_100_in_0_7 <= load_idx_oh_0(7);
	TEMP_100_in_0_8 <= '0';
	TEMP_100_in_0_9 <= load_idx_oh_0(9);
	TEMP_101_res_0 <= TEMP_100_in_0_0 or TEMP_100_in_0_8;
	TEMP_101_res_1 <= TEMP_100_in_0_1 or TEMP_100_in_0_9;
	TEMP_101_res_2 <= TEMP_100_in_0_2;
	TEMP_101_res_3 <= TEMP_100_in_0_3;
	TEMP_101_res_4 <= TEMP_100_in_0_4;
	TEMP_101_res_5 <= TEMP_100_in_0_5;
	TEMP_101_res_6 <= TEMP_100_in_0_6;
	TEMP_101_res_7 <= TEMP_100_in_0_7;
	-- Layer End
	TEMP_102_res_0 <= TEMP_101_res_0 or TEMP_101_res_4;
	TEMP_102_res_1 <= TEMP_101_res_1 or TEMP_101_res_5;
	TEMP_102_res_2 <= TEMP_101_res_2 or TEMP_101_res_6;
	TEMP_102_res_3 <= TEMP_101_res_3 or TEMP_101_res_7;
	-- Layer End
	TEMP_103_res_0 <= TEMP_102_res_0 or TEMP_102_res_2;
	TEMP_103_res_1 <= TEMP_102_res_1 or TEMP_102_res_3;
	-- Layer End
	TEMP_100_out_0 <= TEMP_103_res_0 or TEMP_103_res_1;
	rreq_id_0_o(0) <= TEMP_100_out_0;
	TEMP_103_in_1_0 <= '0';
	TEMP_103_in_1_1 <= '0';
	TEMP_103_in_1_2 <= load_idx_oh_0(2);
	TEMP_103_in_1_3 <= load_idx_oh_0(3);
	TEMP_103_in_1_4 <= '0';
	TEMP_103_in_1_5 <= '0';
	TEMP_103_in_1_6 <= load_idx_oh_0(6);
	TEMP_103_in_1_7 <= load_idx_oh_0(7);
	TEMP_103_in_1_8 <= '0';
	TEMP_103_in_1_9 <= '0';
	TEMP_104_res_0 <= TEMP_103_in_1_0 or TEMP_103_in_1_8;
	TEMP_104_res_1 <= TEMP_103_in_1_1 or TEMP_103_in_1_9;
	TEMP_104_res_2 <= TEMP_103_in_1_2;
	TEMP_104_res_3 <= TEMP_103_in_1_3;
	TEMP_104_res_4 <= TEMP_103_in_1_4;
	TEMP_104_res_5 <= TEMP_103_in_1_5;
	TEMP_104_res_6 <= TEMP_103_in_1_6;
	TEMP_104_res_7 <= TEMP_103_in_1_7;
	-- Layer End
	TEMP_105_res_0 <= TEMP_104_res_0 or TEMP_104_res_4;
	TEMP_105_res_1 <= TEMP_104_res_1 or TEMP_104_res_5;
	TEMP_105_res_2 <= TEMP_104_res_2 or TEMP_104_res_6;
	TEMP_105_res_3 <= TEMP_104_res_3 or TEMP_104_res_7;
	-- Layer End
	TEMP_106_res_0 <= TEMP_105_res_0 or TEMP_105_res_2;
	TEMP_106_res_1 <= TEMP_105_res_1 or TEMP_105_res_3;
	-- Layer End
	TEMP_103_out_1 <= TEMP_106_res_0 or TEMP_106_res_1;
	rreq_id_0_o(1) <= TEMP_103_out_1;
	TEMP_106_in_2_0 <= '0';
	TEMP_106_in_2_1 <= '0';
	TEMP_106_in_2_2 <= '0';
	TEMP_106_in_2_3 <= '0';
	TEMP_106_in_2_4 <= load_idx_oh_0(4);
	TEMP_106_in_2_5 <= load_idx_oh_0(5);
	TEMP_106_in_2_6 <= load_idx_oh_0(6);
	TEMP_106_in_2_7 <= load_idx_oh_0(7);
	TEMP_106_in_2_8 <= '0';
	TEMP_106_in_2_9 <= '0';
	TEMP_107_res_0 <= TEMP_106_in_2_0 or TEMP_106_in_2_8;
	TEMP_107_res_1 <= TEMP_106_in_2_1 or TEMP_106_in_2_9;
	TEMP_107_res_2 <= TEMP_106_in_2_2;
	TEMP_107_res_3 <= TEMP_106_in_2_3;
	TEMP_107_res_4 <= TEMP_106_in_2_4;
	TEMP_107_res_5 <= TEMP_106_in_2_5;
	TEMP_107_res_6 <= TEMP_106_in_2_6;
	TEMP_107_res_7 <= TEMP_106_in_2_7;
	-- Layer End
	TEMP_108_res_0 <= TEMP_107_res_0 or TEMP_107_res_4;
	TEMP_108_res_1 <= TEMP_107_res_1 or TEMP_107_res_5;
	TEMP_108_res_2 <= TEMP_107_res_2 or TEMP_107_res_6;
	TEMP_108_res_3 <= TEMP_107_res_3 or TEMP_107_res_7;
	-- Layer End
	TEMP_109_res_0 <= TEMP_108_res_0 or TEMP_108_res_2;
	TEMP_109_res_1 <= TEMP_108_res_1 or TEMP_108_res_3;
	-- Layer End
	TEMP_106_out_2 <= TEMP_109_res_0 or TEMP_109_res_1;
	rreq_id_0_o(2) <= TEMP_106_out_2;
	TEMP_109_in_3_0 <= '0';
	TEMP_109_in_3_1 <= '0';
	TEMP_109_in_3_2 <= '0';
	TEMP_109_in_3_3 <= '0';
	TEMP_109_in_3_4 <= '0';
	TEMP_109_in_3_5 <= '0';
	TEMP_109_in_3_6 <= '0';
	TEMP_109_in_3_7 <= '0';
	TEMP_109_in_3_8 <= load_idx_oh_0(8);
	TEMP_109_in_3_9 <= load_idx_oh_0(9);
	TEMP_110_res_0 <= TEMP_109_in_3_0 or TEMP_109_in_3_8;
	TEMP_110_res_1 <= TEMP_109_in_3_1 or TEMP_109_in_3_9;
	TEMP_110_res_2 <= TEMP_109_in_3_2;
	TEMP_110_res_3 <= TEMP_109_in_3_3;
	TEMP_110_res_4 <= TEMP_109_in_3_4;
	TEMP_110_res_5 <= TEMP_109_in_3_5;
	TEMP_110_res_6 <= TEMP_109_in_3_6;
	TEMP_110_res_7 <= TEMP_109_in_3_7;
	-- Layer End
	TEMP_111_res_0 <= TEMP_110_res_0 or TEMP_110_res_4;
	TEMP_111_res_1 <= TEMP_110_res_1 or TEMP_110_res_5;
	TEMP_111_res_2 <= TEMP_110_res_2 or TEMP_110_res_6;
	TEMP_111_res_3 <= TEMP_110_res_3 or TEMP_110_res_7;
	-- Layer End
	TEMP_112_res_0 <= TEMP_111_res_0 or TEMP_111_res_2;
	TEMP_112_res_1 <= TEMP_111_res_1 or TEMP_111_res_3;
	-- Layer End
	TEMP_109_out_3 <= TEMP_112_res_0 or TEMP_112_res_1;
	rreq_id_0_o(3) <= TEMP_109_out_3;
	-- One-Hot To Bits End

	-- Mux1H Begin
	-- Mux1H(rreq_addr_0, ldq_addr, load_idx_oh_0)
	TEMP_113_mux_0 <= ldq_addr_0_q when load_idx_oh_0(0) = '1' else "0000000000";
	TEMP_113_mux_1 <= ldq_addr_1_q when load_idx_oh_0(1) = '1' else "0000000000";
	TEMP_113_mux_2 <= ldq_addr_2_q when load_idx_oh_0(2) = '1' else "0000000000";
	TEMP_113_mux_3 <= ldq_addr_3_q when load_idx_oh_0(3) = '1' else "0000000000";
	TEMP_113_mux_4 <= ldq_addr_4_q when load_idx_oh_0(4) = '1' else "0000000000";
	TEMP_113_mux_5 <= ldq_addr_5_q when load_idx_oh_0(5) = '1' else "0000000000";
	TEMP_113_mux_6 <= ldq_addr_6_q when load_idx_oh_0(6) = '1' else "0000000000";
	TEMP_113_mux_7 <= ldq_addr_7_q when load_idx_oh_0(7) = '1' else "0000000000";
	TEMP_113_mux_8 <= ldq_addr_8_q when load_idx_oh_0(8) = '1' else "0000000000";
	TEMP_113_mux_9 <= ldq_addr_9_q when load_idx_oh_0(9) = '1' else "0000000000";
	TEMP_114_res_0 <= TEMP_113_mux_0 or TEMP_113_mux_8;
	TEMP_114_res_1 <= TEMP_113_mux_1 or TEMP_113_mux_9;
	TEMP_114_res_2 <= TEMP_113_mux_2;
	TEMP_114_res_3 <= TEMP_113_mux_3;
	TEMP_114_res_4 <= TEMP_113_mux_4;
	TEMP_114_res_5 <= TEMP_113_mux_5;
	TEMP_114_res_6 <= TEMP_113_mux_6;
	TEMP_114_res_7 <= TEMP_113_mux_7;
	-- Layer End
	TEMP_115_res_0 <= TEMP_114_res_0 or TEMP_114_res_4;
	TEMP_115_res_1 <= TEMP_114_res_1 or TEMP_114_res_5;
	TEMP_115_res_2 <= TEMP_114_res_2 or TEMP_114_res_6;
	TEMP_115_res_3 <= TEMP_114_res_3 or TEMP_114_res_7;
	-- Layer End
	TEMP_116_res_0 <= TEMP_115_res_0 or TEMP_115_res_2;
	TEMP_116_res_1 <= TEMP_115_res_1 or TEMP_115_res_3;
	-- Layer End
	rreq_addr_0_o <= TEMP_116_res_0 or TEMP_116_res_1;
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

	ldq_issue_set_vec_2(0) <= ( load_idx_oh_0(2) and rreq_ready_0_i and load_en_0 ) or bypass_en_2;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_2, ldq_issue_set_vec_2, or)
	ldq_issue_set_2 <= ldq_issue_set_vec_2(0);
	-- Reduction End

	ldq_issue_set_vec_3(0) <= ( load_idx_oh_0(3) and rreq_ready_0_i and load_en_0 ) or bypass_en_3;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_3, ldq_issue_set_vec_3, or)
	ldq_issue_set_3 <= ldq_issue_set_vec_3(0);
	-- Reduction End

	ldq_issue_set_vec_4(0) <= ( load_idx_oh_0(4) and rreq_ready_0_i and load_en_0 ) or bypass_en_4;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_4, ldq_issue_set_vec_4, or)
	ldq_issue_set_4 <= ldq_issue_set_vec_4(0);
	-- Reduction End

	ldq_issue_set_vec_5(0) <= ( load_idx_oh_0(5) and rreq_ready_0_i and load_en_0 ) or bypass_en_5;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_5, ldq_issue_set_vec_5, or)
	ldq_issue_set_5 <= ldq_issue_set_vec_5(0);
	-- Reduction End

	ldq_issue_set_vec_6(0) <= ( load_idx_oh_0(6) and rreq_ready_0_i and load_en_0 ) or bypass_en_6;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_6, ldq_issue_set_vec_6, or)
	ldq_issue_set_6 <= ldq_issue_set_vec_6(0);
	-- Reduction End

	ldq_issue_set_vec_7(0) <= ( load_idx_oh_0(7) and rreq_ready_0_i and load_en_0 ) or bypass_en_7;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_7, ldq_issue_set_vec_7, or)
	ldq_issue_set_7 <= ldq_issue_set_vec_7(0);
	-- Reduction End

	ldq_issue_set_vec_8(0) <= ( load_idx_oh_0(8) and rreq_ready_0_i and load_en_0 ) or bypass_en_8;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_8, ldq_issue_set_vec_8, or)
	ldq_issue_set_8 <= ldq_issue_set_vec_8(0);
	-- Reduction End

	ldq_issue_set_vec_9(0) <= ( load_idx_oh_0(9) and rreq_ready_0_i and load_en_0 ) or bypass_en_9;
	-- Reduction Begin
	-- Reduce(ldq_issue_set_9, ldq_issue_set_vec_9, or)
	ldq_issue_set_9 <= ldq_issue_set_vec_9(0);
	-- Reduction End

	wreq_valid_0_o <= store_en;
	wreq_id_0_o <= "0000";
	-- MuxLookUp Begin
	-- MuxLookUp(wreq_addr_0, stq_addr, store_idx)
	wreq_addr_0_o <= 
	stq_addr_0_q when (store_idx = "0000") else
	stq_addr_1_q when (store_idx = "0001") else
	stq_addr_2_q when (store_idx = "0010") else
	stq_addr_3_q when (store_idx = "0011") else
	stq_addr_4_q when (store_idx = "0100") else
	stq_addr_5_q when (store_idx = "0101") else
	stq_addr_6_q when (store_idx = "0110") else
	stq_addr_7_q when (store_idx = "0111") else
	stq_addr_8_q when (store_idx = "1000") else
	stq_addr_9_q when (store_idx = "1001") else
	"0000000000";
	-- MuxLookUp End

	-- MuxLookUp Begin
	-- MuxLookUp(wreq_data_0, stq_data, store_idx)
	wreq_data_0_o <= 
	stq_data_0_q when (store_idx = "0000") else
	stq_data_1_q when (store_idx = "0001") else
	stq_data_2_q when (store_idx = "0010") else
	stq_data_3_q when (store_idx = "0011") else
	stq_data_4_q when (store_idx = "0100") else
	stq_data_5_q when (store_idx = "0101") else
	stq_data_6_q when (store_idx = "0110") else
	stq_data_7_q when (store_idx = "0111") else
	stq_data_8_q when (store_idx = "1000") else
	stq_data_9_q when (store_idx = "1001") else
	"00000000000000000000000000000000";
	-- MuxLookUp End

	stq_issue_en <= store_en and wreq_ready_0_i;
	read_idx_oh_0_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_0, rresp_data, read_idx_oh_0)
	TEMP_117_mux_0 <= rresp_data_0_i when read_idx_oh_0_0 = '1' else "00000000000000000000000000000000";
	read_data_0 <= TEMP_117_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_0, read_idx_oh_0, or)
	read_valid_0 <= read_idx_oh_0_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_0, stq_data, bypass_idx_oh_0)
	TEMP_118_mux_0 <= stq_data_0_q when bypass_idx_oh_0(0) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_1 <= stq_data_1_q when bypass_idx_oh_0(1) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_2 <= stq_data_2_q when bypass_idx_oh_0(2) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_3 <= stq_data_3_q when bypass_idx_oh_0(3) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_4 <= stq_data_4_q when bypass_idx_oh_0(4) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_5 <= stq_data_5_q when bypass_idx_oh_0(5) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_6 <= stq_data_6_q when bypass_idx_oh_0(6) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_7 <= stq_data_7_q when bypass_idx_oh_0(7) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_8 <= stq_data_8_q when bypass_idx_oh_0(8) = '1' else "00000000000000000000000000000000";
	TEMP_118_mux_9 <= stq_data_9_q when bypass_idx_oh_0(9) = '1' else "00000000000000000000000000000000";
	TEMP_119_res_0 <= TEMP_118_mux_0 or TEMP_118_mux_8;
	TEMP_119_res_1 <= TEMP_118_mux_1 or TEMP_118_mux_9;
	TEMP_119_res_2 <= TEMP_118_mux_2;
	TEMP_119_res_3 <= TEMP_118_mux_3;
	TEMP_119_res_4 <= TEMP_118_mux_4;
	TEMP_119_res_5 <= TEMP_118_mux_5;
	TEMP_119_res_6 <= TEMP_118_mux_6;
	TEMP_119_res_7 <= TEMP_118_mux_7;
	-- Layer End
	TEMP_120_res_0 <= TEMP_119_res_0 or TEMP_119_res_4;
	TEMP_120_res_1 <= TEMP_119_res_1 or TEMP_119_res_5;
	TEMP_120_res_2 <= TEMP_119_res_2 or TEMP_119_res_6;
	TEMP_120_res_3 <= TEMP_119_res_3 or TEMP_119_res_7;
	-- Layer End
	TEMP_121_res_0 <= TEMP_120_res_0 or TEMP_120_res_2;
	TEMP_121_res_1 <= TEMP_120_res_1 or TEMP_120_res_3;
	-- Layer End
	bypass_data_0 <= TEMP_121_res_0 or TEMP_121_res_1;
	-- Mux1H End

	ldq_data_0_d <= read_data_0 or bypass_data_0;
	ldq_data_wen_0 <= bypass_en_0 or read_valid_0;
	read_idx_oh_1_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_1, rresp_data, read_idx_oh_1)
	TEMP_122_mux_0 <= rresp_data_0_i when read_idx_oh_1_0 = '1' else "00000000000000000000000000000000";
	read_data_1 <= TEMP_122_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_1, read_idx_oh_1, or)
	read_valid_1 <= read_idx_oh_1_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_1, stq_data, bypass_idx_oh_1)
	TEMP_123_mux_0 <= stq_data_0_q when bypass_idx_oh_1(0) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_1 <= stq_data_1_q when bypass_idx_oh_1(1) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_2 <= stq_data_2_q when bypass_idx_oh_1(2) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_3 <= stq_data_3_q when bypass_idx_oh_1(3) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_4 <= stq_data_4_q when bypass_idx_oh_1(4) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_5 <= stq_data_5_q when bypass_idx_oh_1(5) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_6 <= stq_data_6_q when bypass_idx_oh_1(6) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_7 <= stq_data_7_q when bypass_idx_oh_1(7) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_8 <= stq_data_8_q when bypass_idx_oh_1(8) = '1' else "00000000000000000000000000000000";
	TEMP_123_mux_9 <= stq_data_9_q when bypass_idx_oh_1(9) = '1' else "00000000000000000000000000000000";
	TEMP_124_res_0 <= TEMP_123_mux_0 or TEMP_123_mux_8;
	TEMP_124_res_1 <= TEMP_123_mux_1 or TEMP_123_mux_9;
	TEMP_124_res_2 <= TEMP_123_mux_2;
	TEMP_124_res_3 <= TEMP_123_mux_3;
	TEMP_124_res_4 <= TEMP_123_mux_4;
	TEMP_124_res_5 <= TEMP_123_mux_5;
	TEMP_124_res_6 <= TEMP_123_mux_6;
	TEMP_124_res_7 <= TEMP_123_mux_7;
	-- Layer End
	TEMP_125_res_0 <= TEMP_124_res_0 or TEMP_124_res_4;
	TEMP_125_res_1 <= TEMP_124_res_1 or TEMP_124_res_5;
	TEMP_125_res_2 <= TEMP_124_res_2 or TEMP_124_res_6;
	TEMP_125_res_3 <= TEMP_124_res_3 or TEMP_124_res_7;
	-- Layer End
	TEMP_126_res_0 <= TEMP_125_res_0 or TEMP_125_res_2;
	TEMP_126_res_1 <= TEMP_125_res_1 or TEMP_125_res_3;
	-- Layer End
	bypass_data_1 <= TEMP_126_res_0 or TEMP_126_res_1;
	-- Mux1H End

	ldq_data_1_d <= read_data_1 or bypass_data_1;
	ldq_data_wen_1 <= bypass_en_1 or read_valid_1;
	read_idx_oh_2_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0010" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_2, rresp_data, read_idx_oh_2)
	TEMP_127_mux_0 <= rresp_data_0_i when read_idx_oh_2_0 = '1' else "00000000000000000000000000000000";
	read_data_2 <= TEMP_127_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_2, read_idx_oh_2, or)
	read_valid_2 <= read_idx_oh_2_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_2, stq_data, bypass_idx_oh_2)
	TEMP_128_mux_0 <= stq_data_0_q when bypass_idx_oh_2(0) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_1 <= stq_data_1_q when bypass_idx_oh_2(1) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_2 <= stq_data_2_q when bypass_idx_oh_2(2) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_3 <= stq_data_3_q when bypass_idx_oh_2(3) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_4 <= stq_data_4_q when bypass_idx_oh_2(4) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_5 <= stq_data_5_q when bypass_idx_oh_2(5) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_6 <= stq_data_6_q when bypass_idx_oh_2(6) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_7 <= stq_data_7_q when bypass_idx_oh_2(7) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_8 <= stq_data_8_q when bypass_idx_oh_2(8) = '1' else "00000000000000000000000000000000";
	TEMP_128_mux_9 <= stq_data_9_q when bypass_idx_oh_2(9) = '1' else "00000000000000000000000000000000";
	TEMP_129_res_0 <= TEMP_128_mux_0 or TEMP_128_mux_8;
	TEMP_129_res_1 <= TEMP_128_mux_1 or TEMP_128_mux_9;
	TEMP_129_res_2 <= TEMP_128_mux_2;
	TEMP_129_res_3 <= TEMP_128_mux_3;
	TEMP_129_res_4 <= TEMP_128_mux_4;
	TEMP_129_res_5 <= TEMP_128_mux_5;
	TEMP_129_res_6 <= TEMP_128_mux_6;
	TEMP_129_res_7 <= TEMP_128_mux_7;
	-- Layer End
	TEMP_130_res_0 <= TEMP_129_res_0 or TEMP_129_res_4;
	TEMP_130_res_1 <= TEMP_129_res_1 or TEMP_129_res_5;
	TEMP_130_res_2 <= TEMP_129_res_2 or TEMP_129_res_6;
	TEMP_130_res_3 <= TEMP_129_res_3 or TEMP_129_res_7;
	-- Layer End
	TEMP_131_res_0 <= TEMP_130_res_0 or TEMP_130_res_2;
	TEMP_131_res_1 <= TEMP_130_res_1 or TEMP_130_res_3;
	-- Layer End
	bypass_data_2 <= TEMP_131_res_0 or TEMP_131_res_1;
	-- Mux1H End

	ldq_data_2_d <= read_data_2 or bypass_data_2;
	ldq_data_wen_2 <= bypass_en_2 or read_valid_2;
	read_idx_oh_3_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0011" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_3, rresp_data, read_idx_oh_3)
	TEMP_132_mux_0 <= rresp_data_0_i when read_idx_oh_3_0 = '1' else "00000000000000000000000000000000";
	read_data_3 <= TEMP_132_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_3, read_idx_oh_3, or)
	read_valid_3 <= read_idx_oh_3_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_3, stq_data, bypass_idx_oh_3)
	TEMP_133_mux_0 <= stq_data_0_q when bypass_idx_oh_3(0) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_1 <= stq_data_1_q when bypass_idx_oh_3(1) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_2 <= stq_data_2_q when bypass_idx_oh_3(2) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_3 <= stq_data_3_q when bypass_idx_oh_3(3) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_4 <= stq_data_4_q when bypass_idx_oh_3(4) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_5 <= stq_data_5_q when bypass_idx_oh_3(5) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_6 <= stq_data_6_q when bypass_idx_oh_3(6) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_7 <= stq_data_7_q when bypass_idx_oh_3(7) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_8 <= stq_data_8_q when bypass_idx_oh_3(8) = '1' else "00000000000000000000000000000000";
	TEMP_133_mux_9 <= stq_data_9_q when bypass_idx_oh_3(9) = '1' else "00000000000000000000000000000000";
	TEMP_134_res_0 <= TEMP_133_mux_0 or TEMP_133_mux_8;
	TEMP_134_res_1 <= TEMP_133_mux_1 or TEMP_133_mux_9;
	TEMP_134_res_2 <= TEMP_133_mux_2;
	TEMP_134_res_3 <= TEMP_133_mux_3;
	TEMP_134_res_4 <= TEMP_133_mux_4;
	TEMP_134_res_5 <= TEMP_133_mux_5;
	TEMP_134_res_6 <= TEMP_133_mux_6;
	TEMP_134_res_7 <= TEMP_133_mux_7;
	-- Layer End
	TEMP_135_res_0 <= TEMP_134_res_0 or TEMP_134_res_4;
	TEMP_135_res_1 <= TEMP_134_res_1 or TEMP_134_res_5;
	TEMP_135_res_2 <= TEMP_134_res_2 or TEMP_134_res_6;
	TEMP_135_res_3 <= TEMP_134_res_3 or TEMP_134_res_7;
	-- Layer End
	TEMP_136_res_0 <= TEMP_135_res_0 or TEMP_135_res_2;
	TEMP_136_res_1 <= TEMP_135_res_1 or TEMP_135_res_3;
	-- Layer End
	bypass_data_3 <= TEMP_136_res_0 or TEMP_136_res_1;
	-- Mux1H End

	ldq_data_3_d <= read_data_3 or bypass_data_3;
	ldq_data_wen_3 <= bypass_en_3 or read_valid_3;
	read_idx_oh_4_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0100" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_4, rresp_data, read_idx_oh_4)
	TEMP_137_mux_0 <= rresp_data_0_i when read_idx_oh_4_0 = '1' else "00000000000000000000000000000000";
	read_data_4 <= TEMP_137_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_4, read_idx_oh_4, or)
	read_valid_4 <= read_idx_oh_4_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_4, stq_data, bypass_idx_oh_4)
	TEMP_138_mux_0 <= stq_data_0_q when bypass_idx_oh_4(0) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_1 <= stq_data_1_q when bypass_idx_oh_4(1) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_2 <= stq_data_2_q when bypass_idx_oh_4(2) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_3 <= stq_data_3_q when bypass_idx_oh_4(3) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_4 <= stq_data_4_q when bypass_idx_oh_4(4) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_5 <= stq_data_5_q when bypass_idx_oh_4(5) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_6 <= stq_data_6_q when bypass_idx_oh_4(6) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_7 <= stq_data_7_q when bypass_idx_oh_4(7) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_8 <= stq_data_8_q when bypass_idx_oh_4(8) = '1' else "00000000000000000000000000000000";
	TEMP_138_mux_9 <= stq_data_9_q when bypass_idx_oh_4(9) = '1' else "00000000000000000000000000000000";
	TEMP_139_res_0 <= TEMP_138_mux_0 or TEMP_138_mux_8;
	TEMP_139_res_1 <= TEMP_138_mux_1 or TEMP_138_mux_9;
	TEMP_139_res_2 <= TEMP_138_mux_2;
	TEMP_139_res_3 <= TEMP_138_mux_3;
	TEMP_139_res_4 <= TEMP_138_mux_4;
	TEMP_139_res_5 <= TEMP_138_mux_5;
	TEMP_139_res_6 <= TEMP_138_mux_6;
	TEMP_139_res_7 <= TEMP_138_mux_7;
	-- Layer End
	TEMP_140_res_0 <= TEMP_139_res_0 or TEMP_139_res_4;
	TEMP_140_res_1 <= TEMP_139_res_1 or TEMP_139_res_5;
	TEMP_140_res_2 <= TEMP_139_res_2 or TEMP_139_res_6;
	TEMP_140_res_3 <= TEMP_139_res_3 or TEMP_139_res_7;
	-- Layer End
	TEMP_141_res_0 <= TEMP_140_res_0 or TEMP_140_res_2;
	TEMP_141_res_1 <= TEMP_140_res_1 or TEMP_140_res_3;
	-- Layer End
	bypass_data_4 <= TEMP_141_res_0 or TEMP_141_res_1;
	-- Mux1H End

	ldq_data_4_d <= read_data_4 or bypass_data_4;
	ldq_data_wen_4 <= bypass_en_4 or read_valid_4;
	read_idx_oh_5_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0101" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_5, rresp_data, read_idx_oh_5)
	TEMP_142_mux_0 <= rresp_data_0_i when read_idx_oh_5_0 = '1' else "00000000000000000000000000000000";
	read_data_5 <= TEMP_142_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_5, read_idx_oh_5, or)
	read_valid_5 <= read_idx_oh_5_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_5, stq_data, bypass_idx_oh_5)
	TEMP_143_mux_0 <= stq_data_0_q when bypass_idx_oh_5(0) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_1 <= stq_data_1_q when bypass_idx_oh_5(1) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_2 <= stq_data_2_q when bypass_idx_oh_5(2) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_3 <= stq_data_3_q when bypass_idx_oh_5(3) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_4 <= stq_data_4_q when bypass_idx_oh_5(4) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_5 <= stq_data_5_q when bypass_idx_oh_5(5) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_6 <= stq_data_6_q when bypass_idx_oh_5(6) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_7 <= stq_data_7_q when bypass_idx_oh_5(7) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_8 <= stq_data_8_q when bypass_idx_oh_5(8) = '1' else "00000000000000000000000000000000";
	TEMP_143_mux_9 <= stq_data_9_q when bypass_idx_oh_5(9) = '1' else "00000000000000000000000000000000";
	TEMP_144_res_0 <= TEMP_143_mux_0 or TEMP_143_mux_8;
	TEMP_144_res_1 <= TEMP_143_mux_1 or TEMP_143_mux_9;
	TEMP_144_res_2 <= TEMP_143_mux_2;
	TEMP_144_res_3 <= TEMP_143_mux_3;
	TEMP_144_res_4 <= TEMP_143_mux_4;
	TEMP_144_res_5 <= TEMP_143_mux_5;
	TEMP_144_res_6 <= TEMP_143_mux_6;
	TEMP_144_res_7 <= TEMP_143_mux_7;
	-- Layer End
	TEMP_145_res_0 <= TEMP_144_res_0 or TEMP_144_res_4;
	TEMP_145_res_1 <= TEMP_144_res_1 or TEMP_144_res_5;
	TEMP_145_res_2 <= TEMP_144_res_2 or TEMP_144_res_6;
	TEMP_145_res_3 <= TEMP_144_res_3 or TEMP_144_res_7;
	-- Layer End
	TEMP_146_res_0 <= TEMP_145_res_0 or TEMP_145_res_2;
	TEMP_146_res_1 <= TEMP_145_res_1 or TEMP_145_res_3;
	-- Layer End
	bypass_data_5 <= TEMP_146_res_0 or TEMP_146_res_1;
	-- Mux1H End

	ldq_data_5_d <= read_data_5 or bypass_data_5;
	ldq_data_wen_5 <= bypass_en_5 or read_valid_5;
	read_idx_oh_6_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0110" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_6, rresp_data, read_idx_oh_6)
	TEMP_147_mux_0 <= rresp_data_0_i when read_idx_oh_6_0 = '1' else "00000000000000000000000000000000";
	read_data_6 <= TEMP_147_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_6, read_idx_oh_6, or)
	read_valid_6 <= read_idx_oh_6_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_6, stq_data, bypass_idx_oh_6)
	TEMP_148_mux_0 <= stq_data_0_q when bypass_idx_oh_6(0) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_1 <= stq_data_1_q when bypass_idx_oh_6(1) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_2 <= stq_data_2_q when bypass_idx_oh_6(2) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_3 <= stq_data_3_q when bypass_idx_oh_6(3) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_4 <= stq_data_4_q when bypass_idx_oh_6(4) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_5 <= stq_data_5_q when bypass_idx_oh_6(5) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_6 <= stq_data_6_q when bypass_idx_oh_6(6) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_7 <= stq_data_7_q when bypass_idx_oh_6(7) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_8 <= stq_data_8_q when bypass_idx_oh_6(8) = '1' else "00000000000000000000000000000000";
	TEMP_148_mux_9 <= stq_data_9_q when bypass_idx_oh_6(9) = '1' else "00000000000000000000000000000000";
	TEMP_149_res_0 <= TEMP_148_mux_0 or TEMP_148_mux_8;
	TEMP_149_res_1 <= TEMP_148_mux_1 or TEMP_148_mux_9;
	TEMP_149_res_2 <= TEMP_148_mux_2;
	TEMP_149_res_3 <= TEMP_148_mux_3;
	TEMP_149_res_4 <= TEMP_148_mux_4;
	TEMP_149_res_5 <= TEMP_148_mux_5;
	TEMP_149_res_6 <= TEMP_148_mux_6;
	TEMP_149_res_7 <= TEMP_148_mux_7;
	-- Layer End
	TEMP_150_res_0 <= TEMP_149_res_0 or TEMP_149_res_4;
	TEMP_150_res_1 <= TEMP_149_res_1 or TEMP_149_res_5;
	TEMP_150_res_2 <= TEMP_149_res_2 or TEMP_149_res_6;
	TEMP_150_res_3 <= TEMP_149_res_3 or TEMP_149_res_7;
	-- Layer End
	TEMP_151_res_0 <= TEMP_150_res_0 or TEMP_150_res_2;
	TEMP_151_res_1 <= TEMP_150_res_1 or TEMP_150_res_3;
	-- Layer End
	bypass_data_6 <= TEMP_151_res_0 or TEMP_151_res_1;
	-- Mux1H End

	ldq_data_6_d <= read_data_6 or bypass_data_6;
	ldq_data_wen_6 <= bypass_en_6 or read_valid_6;
	read_idx_oh_7_0 <= rresp_valid_0_i when ( rresp_id_0_i = "0111" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_7, rresp_data, read_idx_oh_7)
	TEMP_152_mux_0 <= rresp_data_0_i when read_idx_oh_7_0 = '1' else "00000000000000000000000000000000";
	read_data_7 <= TEMP_152_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_7, read_idx_oh_7, or)
	read_valid_7 <= read_idx_oh_7_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_7, stq_data, bypass_idx_oh_7)
	TEMP_153_mux_0 <= stq_data_0_q when bypass_idx_oh_7(0) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_1 <= stq_data_1_q when bypass_idx_oh_7(1) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_2 <= stq_data_2_q when bypass_idx_oh_7(2) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_3 <= stq_data_3_q when bypass_idx_oh_7(3) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_4 <= stq_data_4_q when bypass_idx_oh_7(4) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_5 <= stq_data_5_q when bypass_idx_oh_7(5) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_6 <= stq_data_6_q when bypass_idx_oh_7(6) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_7 <= stq_data_7_q when bypass_idx_oh_7(7) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_8 <= stq_data_8_q when bypass_idx_oh_7(8) = '1' else "00000000000000000000000000000000";
	TEMP_153_mux_9 <= stq_data_9_q when bypass_idx_oh_7(9) = '1' else "00000000000000000000000000000000";
	TEMP_154_res_0 <= TEMP_153_mux_0 or TEMP_153_mux_8;
	TEMP_154_res_1 <= TEMP_153_mux_1 or TEMP_153_mux_9;
	TEMP_154_res_2 <= TEMP_153_mux_2;
	TEMP_154_res_3 <= TEMP_153_mux_3;
	TEMP_154_res_4 <= TEMP_153_mux_4;
	TEMP_154_res_5 <= TEMP_153_mux_5;
	TEMP_154_res_6 <= TEMP_153_mux_6;
	TEMP_154_res_7 <= TEMP_153_mux_7;
	-- Layer End
	TEMP_155_res_0 <= TEMP_154_res_0 or TEMP_154_res_4;
	TEMP_155_res_1 <= TEMP_154_res_1 or TEMP_154_res_5;
	TEMP_155_res_2 <= TEMP_154_res_2 or TEMP_154_res_6;
	TEMP_155_res_3 <= TEMP_154_res_3 or TEMP_154_res_7;
	-- Layer End
	TEMP_156_res_0 <= TEMP_155_res_0 or TEMP_155_res_2;
	TEMP_156_res_1 <= TEMP_155_res_1 or TEMP_155_res_3;
	-- Layer End
	bypass_data_7 <= TEMP_156_res_0 or TEMP_156_res_1;
	-- Mux1H End

	ldq_data_7_d <= read_data_7 or bypass_data_7;
	ldq_data_wen_7 <= bypass_en_7 or read_valid_7;
	read_idx_oh_8_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1000" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_8, rresp_data, read_idx_oh_8)
	TEMP_157_mux_0 <= rresp_data_0_i when read_idx_oh_8_0 = '1' else "00000000000000000000000000000000";
	read_data_8 <= TEMP_157_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_8, read_idx_oh_8, or)
	read_valid_8 <= read_idx_oh_8_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_8, stq_data, bypass_idx_oh_8)
	TEMP_158_mux_0 <= stq_data_0_q when bypass_idx_oh_8(0) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_1 <= stq_data_1_q when bypass_idx_oh_8(1) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_2 <= stq_data_2_q when bypass_idx_oh_8(2) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_3 <= stq_data_3_q when bypass_idx_oh_8(3) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_4 <= stq_data_4_q when bypass_idx_oh_8(4) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_5 <= stq_data_5_q when bypass_idx_oh_8(5) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_6 <= stq_data_6_q when bypass_idx_oh_8(6) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_7 <= stq_data_7_q when bypass_idx_oh_8(7) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_8 <= stq_data_8_q when bypass_idx_oh_8(8) = '1' else "00000000000000000000000000000000";
	TEMP_158_mux_9 <= stq_data_9_q when bypass_idx_oh_8(9) = '1' else "00000000000000000000000000000000";
	TEMP_159_res_0 <= TEMP_158_mux_0 or TEMP_158_mux_8;
	TEMP_159_res_1 <= TEMP_158_mux_1 or TEMP_158_mux_9;
	TEMP_159_res_2 <= TEMP_158_mux_2;
	TEMP_159_res_3 <= TEMP_158_mux_3;
	TEMP_159_res_4 <= TEMP_158_mux_4;
	TEMP_159_res_5 <= TEMP_158_mux_5;
	TEMP_159_res_6 <= TEMP_158_mux_6;
	TEMP_159_res_7 <= TEMP_158_mux_7;
	-- Layer End
	TEMP_160_res_0 <= TEMP_159_res_0 or TEMP_159_res_4;
	TEMP_160_res_1 <= TEMP_159_res_1 or TEMP_159_res_5;
	TEMP_160_res_2 <= TEMP_159_res_2 or TEMP_159_res_6;
	TEMP_160_res_3 <= TEMP_159_res_3 or TEMP_159_res_7;
	-- Layer End
	TEMP_161_res_0 <= TEMP_160_res_0 or TEMP_160_res_2;
	TEMP_161_res_1 <= TEMP_160_res_1 or TEMP_160_res_3;
	-- Layer End
	bypass_data_8 <= TEMP_161_res_0 or TEMP_161_res_1;
	-- Mux1H End

	ldq_data_8_d <= read_data_8 or bypass_data_8;
	ldq_data_wen_8 <= bypass_en_8 or read_valid_8;
	read_idx_oh_9_0 <= rresp_valid_0_i when ( rresp_id_0_i = "1001" ) else '0';
	-- Mux1H Begin
	-- Mux1H(read_data_9, rresp_data, read_idx_oh_9)
	TEMP_162_mux_0 <= rresp_data_0_i when read_idx_oh_9_0 = '1' else "00000000000000000000000000000000";
	read_data_9 <= TEMP_162_mux_0;
	-- Mux1H End

	-- Reduction Begin
	-- Reduce(read_valid_9, read_idx_oh_9, or)
	read_valid_9 <= read_idx_oh_9_0;
	-- Reduction End

	-- Mux1H Begin
	-- Mux1H(bypass_data_9, stq_data, bypass_idx_oh_9)
	TEMP_163_mux_0 <= stq_data_0_q when bypass_idx_oh_9(0) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_1 <= stq_data_1_q when bypass_idx_oh_9(1) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_2 <= stq_data_2_q when bypass_idx_oh_9(2) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_3 <= stq_data_3_q when bypass_idx_oh_9(3) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_4 <= stq_data_4_q when bypass_idx_oh_9(4) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_5 <= stq_data_5_q when bypass_idx_oh_9(5) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_6 <= stq_data_6_q when bypass_idx_oh_9(6) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_7 <= stq_data_7_q when bypass_idx_oh_9(7) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_8 <= stq_data_8_q when bypass_idx_oh_9(8) = '1' else "00000000000000000000000000000000";
	TEMP_163_mux_9 <= stq_data_9_q when bypass_idx_oh_9(9) = '1' else "00000000000000000000000000000000";
	TEMP_164_res_0 <= TEMP_163_mux_0 or TEMP_163_mux_8;
	TEMP_164_res_1 <= TEMP_163_mux_1 or TEMP_163_mux_9;
	TEMP_164_res_2 <= TEMP_163_mux_2;
	TEMP_164_res_3 <= TEMP_163_mux_3;
	TEMP_164_res_4 <= TEMP_163_mux_4;
	TEMP_164_res_5 <= TEMP_163_mux_5;
	TEMP_164_res_6 <= TEMP_163_mux_6;
	TEMP_164_res_7 <= TEMP_163_mux_7;
	-- Layer End
	TEMP_165_res_0 <= TEMP_164_res_0 or TEMP_164_res_4;
	TEMP_165_res_1 <= TEMP_164_res_1 or TEMP_164_res_5;
	TEMP_165_res_2 <= TEMP_164_res_2 or TEMP_164_res_6;
	TEMP_165_res_3 <= TEMP_164_res_3 or TEMP_164_res_7;
	-- Layer End
	TEMP_166_res_0 <= TEMP_165_res_0 or TEMP_165_res_2;
	TEMP_166_res_1 <= TEMP_165_res_1 or TEMP_165_res_3;
	-- Layer End
	bypass_data_9 <= TEMP_166_res_0 or TEMP_166_res_1;
	-- Mux1H End

	ldq_data_9_d <= read_data_9 or bypass_data_9;
	ldq_data_wen_9 <= bypass_en_9 or read_valid_9;
	rresp_ready_0_o <= '1';
	stq_reset_0 <= wresp_valid_0_i when ( stq_resp_q = "0000" ) else '0';
	stq_reset_1 <= wresp_valid_0_i when ( stq_resp_q = "0001" ) else '0';
	stq_reset_2 <= wresp_valid_0_i when ( stq_resp_q = "0010" ) else '0';
	stq_reset_3 <= wresp_valid_0_i when ( stq_resp_q = "0011" ) else '0';
	stq_reset_4 <= wresp_valid_0_i when ( stq_resp_q = "0100" ) else '0';
	stq_reset_5 <= wresp_valid_0_i when ( stq_resp_q = "0101" ) else '0';
	stq_reset_6 <= wresp_valid_0_i when ( stq_resp_q = "0110" ) else '0';
	stq_reset_7 <= wresp_valid_0_i when ( stq_resp_q = "0111" ) else '0';
	stq_reset_8 <= wresp_valid_0_i when ( stq_resp_q = "1000" ) else '0';
	stq_reset_9 <= wresp_valid_0_i when ( stq_resp_q = "1001" ) else '0';
	stq_resp_en <= wresp_valid_0_i;
	wresp_ready_0_o <= '1';

	process (clk, rst) is
	begin
		if (rst = '1') then
			ldq_alloc_0_q <= '0';
			ldq_alloc_1_q <= '0';
			ldq_alloc_2_q <= '0';
			ldq_alloc_3_q <= '0';
			ldq_alloc_4_q <= '0';
			ldq_alloc_5_q <= '0';
			ldq_alloc_6_q <= '0';
			ldq_alloc_7_q <= '0';
			ldq_alloc_8_q <= '0';
			ldq_alloc_9_q <= '0';
		elsif (rising_edge(clk)) then
			ldq_alloc_0_q <= ldq_alloc_0_d;
			ldq_alloc_1_q <= ldq_alloc_1_d;
			ldq_alloc_2_q <= ldq_alloc_2_d;
			ldq_alloc_3_q <= ldq_alloc_3_d;
			ldq_alloc_4_q <= ldq_alloc_4_d;
			ldq_alloc_5_q <= ldq_alloc_5_d;
			ldq_alloc_6_q <= ldq_alloc_6_d;
			ldq_alloc_7_q <= ldq_alloc_7_d;
			ldq_alloc_8_q <= ldq_alloc_8_d;
			ldq_alloc_9_q <= ldq_alloc_9_d;
		end if;
		if (rising_edge(clk)) then
			ldq_issue_0_q <= ldq_issue_0_d;
			ldq_issue_1_q <= ldq_issue_1_d;
			ldq_issue_2_q <= ldq_issue_2_d;
			ldq_issue_3_q <= ldq_issue_3_d;
			ldq_issue_4_q <= ldq_issue_4_d;
			ldq_issue_5_q <= ldq_issue_5_d;
			ldq_issue_6_q <= ldq_issue_6_d;
			ldq_issue_7_q <= ldq_issue_7_d;
			ldq_issue_8_q <= ldq_issue_8_d;
			ldq_issue_9_q <= ldq_issue_9_d;
		end if;
		if (rising_edge(clk)) then
			ldq_addr_valid_0_q <= ldq_addr_valid_0_d;
			ldq_addr_valid_1_q <= ldq_addr_valid_1_d;
			ldq_addr_valid_2_q <= ldq_addr_valid_2_d;
			ldq_addr_valid_3_q <= ldq_addr_valid_3_d;
			ldq_addr_valid_4_q <= ldq_addr_valid_4_d;
			ldq_addr_valid_5_q <= ldq_addr_valid_5_d;
			ldq_addr_valid_6_q <= ldq_addr_valid_6_d;
			ldq_addr_valid_7_q <= ldq_addr_valid_7_d;
			ldq_addr_valid_8_q <= ldq_addr_valid_8_d;
			ldq_addr_valid_9_q <= ldq_addr_valid_9_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_addr_wen_0 = '1') then
				ldq_addr_0_q <= ldq_addr_0_d;
			end if;
			if (ldq_addr_wen_1 = '1') then
				ldq_addr_1_q <= ldq_addr_1_d;
			end if;
			if (ldq_addr_wen_2 = '1') then
				ldq_addr_2_q <= ldq_addr_2_d;
			end if;
			if (ldq_addr_wen_3 = '1') then
				ldq_addr_3_q <= ldq_addr_3_d;
			end if;
			if (ldq_addr_wen_4 = '1') then
				ldq_addr_4_q <= ldq_addr_4_d;
			end if;
			if (ldq_addr_wen_5 = '1') then
				ldq_addr_5_q <= ldq_addr_5_d;
			end if;
			if (ldq_addr_wen_6 = '1') then
				ldq_addr_6_q <= ldq_addr_6_d;
			end if;
			if (ldq_addr_wen_7 = '1') then
				ldq_addr_7_q <= ldq_addr_7_d;
			end if;
			if (ldq_addr_wen_8 = '1') then
				ldq_addr_8_q <= ldq_addr_8_d;
			end if;
			if (ldq_addr_wen_9 = '1') then
				ldq_addr_9_q <= ldq_addr_9_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			ldq_data_valid_0_q <= ldq_data_valid_0_d;
			ldq_data_valid_1_q <= ldq_data_valid_1_d;
			ldq_data_valid_2_q <= ldq_data_valid_2_d;
			ldq_data_valid_3_q <= ldq_data_valid_3_d;
			ldq_data_valid_4_q <= ldq_data_valid_4_d;
			ldq_data_valid_5_q <= ldq_data_valid_5_d;
			ldq_data_valid_6_q <= ldq_data_valid_6_d;
			ldq_data_valid_7_q <= ldq_data_valid_7_d;
			ldq_data_valid_8_q <= ldq_data_valid_8_d;
			ldq_data_valid_9_q <= ldq_data_valid_9_d;
		end if;
		if (rising_edge(clk)) then
			if (ldq_data_wen_0 = '1') then
				ldq_data_0_q <= ldq_data_0_d;
			end if;
			if (ldq_data_wen_1 = '1') then
				ldq_data_1_q <= ldq_data_1_d;
			end if;
			if (ldq_data_wen_2 = '1') then
				ldq_data_2_q <= ldq_data_2_d;
			end if;
			if (ldq_data_wen_3 = '1') then
				ldq_data_3_q <= ldq_data_3_d;
			end if;
			if (ldq_data_wen_4 = '1') then
				ldq_data_4_q <= ldq_data_4_d;
			end if;
			if (ldq_data_wen_5 = '1') then
				ldq_data_5_q <= ldq_data_5_d;
			end if;
			if (ldq_data_wen_6 = '1') then
				ldq_data_6_q <= ldq_data_6_d;
			end if;
			if (ldq_data_wen_7 = '1') then
				ldq_data_7_q <= ldq_data_7_d;
			end if;
			if (ldq_data_wen_8 = '1') then
				ldq_data_8_q <= ldq_data_8_d;
			end if;
			if (ldq_data_wen_9 = '1') then
				ldq_data_9_q <= ldq_data_9_d;
			end if;
		end if;
		if (rst = '1') then
			stq_alloc_0_q <= '0';
			stq_alloc_1_q <= '0';
			stq_alloc_2_q <= '0';
			stq_alloc_3_q <= '0';
			stq_alloc_4_q <= '0';
			stq_alloc_5_q <= '0';
			stq_alloc_6_q <= '0';
			stq_alloc_7_q <= '0';
			stq_alloc_8_q <= '0';
			stq_alloc_9_q <= '0';
		elsif (rising_edge(clk)) then
			stq_alloc_0_q <= stq_alloc_0_d;
			stq_alloc_1_q <= stq_alloc_1_d;
			stq_alloc_2_q <= stq_alloc_2_d;
			stq_alloc_3_q <= stq_alloc_3_d;
			stq_alloc_4_q <= stq_alloc_4_d;
			stq_alloc_5_q <= stq_alloc_5_d;
			stq_alloc_6_q <= stq_alloc_6_d;
			stq_alloc_7_q <= stq_alloc_7_d;
			stq_alloc_8_q <= stq_alloc_8_d;
			stq_alloc_9_q <= stq_alloc_9_d;
		end if;
		if (rising_edge(clk)) then
			stq_addr_valid_0_q <= stq_addr_valid_0_d;
			stq_addr_valid_1_q <= stq_addr_valid_1_d;
			stq_addr_valid_2_q <= stq_addr_valid_2_d;
			stq_addr_valid_3_q <= stq_addr_valid_3_d;
			stq_addr_valid_4_q <= stq_addr_valid_4_d;
			stq_addr_valid_5_q <= stq_addr_valid_5_d;
			stq_addr_valid_6_q <= stq_addr_valid_6_d;
			stq_addr_valid_7_q <= stq_addr_valid_7_d;
			stq_addr_valid_8_q <= stq_addr_valid_8_d;
			stq_addr_valid_9_q <= stq_addr_valid_9_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_addr_wen_0 = '1') then
				stq_addr_0_q <= stq_addr_0_d;
			end if;
			if (stq_addr_wen_1 = '1') then
				stq_addr_1_q <= stq_addr_1_d;
			end if;
			if (stq_addr_wen_2 = '1') then
				stq_addr_2_q <= stq_addr_2_d;
			end if;
			if (stq_addr_wen_3 = '1') then
				stq_addr_3_q <= stq_addr_3_d;
			end if;
			if (stq_addr_wen_4 = '1') then
				stq_addr_4_q <= stq_addr_4_d;
			end if;
			if (stq_addr_wen_5 = '1') then
				stq_addr_5_q <= stq_addr_5_d;
			end if;
			if (stq_addr_wen_6 = '1') then
				stq_addr_6_q <= stq_addr_6_d;
			end if;
			if (stq_addr_wen_7 = '1') then
				stq_addr_7_q <= stq_addr_7_d;
			end if;
			if (stq_addr_wen_8 = '1') then
				stq_addr_8_q <= stq_addr_8_d;
			end if;
			if (stq_addr_wen_9 = '1') then
				stq_addr_9_q <= stq_addr_9_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			stq_data_valid_0_q <= stq_data_valid_0_d;
			stq_data_valid_1_q <= stq_data_valid_1_d;
			stq_data_valid_2_q <= stq_data_valid_2_d;
			stq_data_valid_3_q <= stq_data_valid_3_d;
			stq_data_valid_4_q <= stq_data_valid_4_d;
			stq_data_valid_5_q <= stq_data_valid_5_d;
			stq_data_valid_6_q <= stq_data_valid_6_d;
			stq_data_valid_7_q <= stq_data_valid_7_d;
			stq_data_valid_8_q <= stq_data_valid_8_d;
			stq_data_valid_9_q <= stq_data_valid_9_d;
		end if;
		if (rising_edge(clk)) then
			if (stq_data_wen_0 = '1') then
				stq_data_0_q <= stq_data_0_d;
			end if;
			if (stq_data_wen_1 = '1') then
				stq_data_1_q <= stq_data_1_d;
			end if;
			if (stq_data_wen_2 = '1') then
				stq_data_2_q <= stq_data_2_d;
			end if;
			if (stq_data_wen_3 = '1') then
				stq_data_3_q <= stq_data_3_d;
			end if;
			if (stq_data_wen_4 = '1') then
				stq_data_4_q <= stq_data_4_d;
			end if;
			if (stq_data_wen_5 = '1') then
				stq_data_5_q <= stq_data_5_d;
			end if;
			if (stq_data_wen_6 = '1') then
				stq_data_6_q <= stq_data_6_d;
			end if;
			if (stq_data_wen_7 = '1') then
				stq_data_7_q <= stq_data_7_d;
			end if;
			if (stq_data_wen_8 = '1') then
				stq_data_8_q <= stq_data_8_d;
			end if;
			if (stq_data_wen_9 = '1') then
				stq_data_9_q <= stq_data_9_d;
			end if;
		end if;
		if (rising_edge(clk)) then
			store_is_older_0_q <= store_is_older_0_d;
			store_is_older_1_q <= store_is_older_1_d;
			store_is_older_2_q <= store_is_older_2_d;
			store_is_older_3_q <= store_is_older_3_d;
			store_is_older_4_q <= store_is_older_4_d;
			store_is_older_5_q <= store_is_older_5_d;
			store_is_older_6_q <= store_is_older_6_d;
			store_is_older_7_q <= store_is_older_7_d;
			store_is_older_8_q <= store_is_older_8_d;
			store_is_older_9_q <= store_is_older_9_d;
		end if;
		if (rst = '1') then
			ldq_tail_q <= "0000";
		elsif (rising_edge(clk)) then
			ldq_tail_q <= ldq_tail_d;
		end if;
		if (rst = '1') then
			ldq_head_q <= "0000";
		elsif (rising_edge(clk)) then
			ldq_head_q <= ldq_head_d;
		end if;
		if (rst = '1') then
			stq_tail_q <= "0000";
		elsif (rising_edge(clk)) then
			stq_tail_q <= stq_tail_d;
		end if;
		if (rst = '1') then
			stq_head_q <= "0000";
		elsif (rising_edge(clk)) then
			stq_head_q <= stq_head_d;
		end if;
		if (rst = '1') then
			stq_issue_q <= "0000";
		elsif (rising_edge(clk)) then
			if (stq_issue_en = '1') then
				stq_issue_q <= stq_issue_d;
			end if;
		end if;
		if (rst = '1') then
			stq_resp_q <= "0000";
		elsif (rising_edge(clk)) then
			if (stq_resp_en = '1') then
				stq_resp_q <= stq_resp_d;
			end if;
		end if;
	end process;
end architecture;
