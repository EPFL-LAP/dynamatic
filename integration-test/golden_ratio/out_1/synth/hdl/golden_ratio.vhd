library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity golden_ratio is
  port (
    x0 : in std_logic_vector(31 downto 0);
    x0_valid : in std_logic;
    x1 : in std_logic_vector(31 downto 0);
    x1_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    x0_ready : out std_logic;
    x1_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of golden_ratio is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(7 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(7 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(0 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(0 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(0 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(7 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(7 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(31 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal passer7_result : std_logic_vector(31 downto 0);
  signal passer7_result_valid : std_logic;
  signal passer7_result_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(31 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(31 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal subf0_result : std_logic_vector(31 downto 0);
  signal subf0_result_valid : std_logic;
  signal subf0_result_ready : std_logic;
  signal absf0_outs : std_logic_vector(31 downto 0);
  signal absf0_outs_valid : std_logic;
  signal absf0_outs_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal passer8_result : std_logic_vector(0 downto 0);
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal spec_v2_repeating_init0_outs : std_logic_vector(0 downto 0);
  signal spec_v2_repeating_init0_outs_valid : std_logic;
  signal spec_v2_repeating_init0_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(0 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2 : std_logic_vector(0 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_3 : std_logic_vector(0 downto 0);
  signal fork2_outs_3_valid : std_logic;
  signal fork2_outs_3_ready : std_logic;
  signal fork2_outs_4 : std_logic_vector(0 downto 0);
  signal fork2_outs_4_valid : std_logic;
  signal fork2_outs_4_ready : std_logic;
  signal fork2_outs_5 : std_logic_vector(0 downto 0);
  signal fork2_outs_5_valid : std_logic;
  signal fork2_outs_5_ready : std_logic;
  signal fork2_outs_6 : std_logic_vector(0 downto 0);
  signal fork2_outs_6_valid : std_logic;
  signal fork2_outs_6_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal init4_outs : std_logic_vector(0 downto 0);
  signal init4_outs_valid : std_logic;
  signal init4_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(0 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal fork3_outs_3 : std_logic_vector(0 downto 0);
  signal fork3_outs_3_valid : std_logic;
  signal fork3_outs_3_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(0 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(0 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal fork16_outs_2 : std_logic_vector(0 downto 0);
  signal fork16_outs_2_valid : std_logic;
  signal fork16_outs_2_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal passer0_result : std_logic_vector(31 downto 0);
  signal passer0_result_valid : std_logic;
  signal passer0_result_ready : std_logic;
  signal buffer15_outs : std_logic_vector(7 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal passer1_result : std_logic_vector(7 downto 0);
  signal passer1_result_valid : std_logic;
  signal passer1_result_ready : std_logic;
  signal passer9_result : std_logic_vector(7 downto 0);
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal buffer4_outs : std_logic_vector(7 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(7 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(7 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(7 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal passer3_result_valid : std_logic;
  signal passer3_result_ready : std_logic;
  signal passer10_result_valid : std_logic;
  signal passer10_result_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(31 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(31 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(31 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal fork19_outs_3 : std_logic_vector(31 downto 0);
  signal fork19_outs_3_valid : std_logic;
  signal fork19_outs_3_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal passer11_result : std_logic_vector(31 downto 0);
  signal passer11_result_valid : std_logic;
  signal passer11_result_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal buffer14_outs : std_logic_vector(7 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(8 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(8 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(7 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(8 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal divf0_result : std_logic_vector(31 downto 0);
  signal divf0_result_valid : std_logic;
  signal divf0_result_ready : std_logic;
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(8 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(8 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal trunci0_outs : std_logic_vector(7 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(0 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(0 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal fork10_outs_2 : std_logic_vector(0 downto 0);
  signal fork10_outs_2_valid : std_logic;
  signal fork10_outs_2_ready : std_logic;
  signal fork10_outs_3 : std_logic_vector(0 downto 0);
  signal fork10_outs_3_valid : std_logic;
  signal fork10_outs_3_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(7 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(7 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(31 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(31 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(31 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(31 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;

begin

  out0 <= cond_br9_falseOut;
  out0_valid <= cond_br9_falseOut_valid;
  cond_br9_falseOut_ready <= out0_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;

  fork0 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready
    );

  constant0 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  extsi3 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => fork1_outs_0,
      index_valid => fork1_outs_0_valid,
      index_ready => fork1_outs_0_ready,
      ins(0) => extsi3_outs,
      ins(1) => cond_br7_trueOut,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork1_outs_1,
      index_valid => fork1_outs_1_valid,
      index_ready => fork1_outs_1_ready,
      ins(0) => x1,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => x1_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => x1_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => fork1_outs_2,
      index_valid => fork1_outs_2_valid,
      index_ready => fork1_outs_2_ready,
      ins(0) => x0,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => x0_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => x0_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready
    );

  buffer2 : entity work.handshake_buffer_0(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  mux3 : entity work.handshake_mux_3(arch)
    port map(
      index => fork3_outs_3,
      index_valid => fork3_outs_3_valid,
      index_ready => fork3_outs_3_ready,
      ins(0) => buffer2_outs,
      ins(1) => passer7_result,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => passer7_result_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => passer7_result_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer0 : entity work.handshake_buffer_1(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  mux4 : entity work.handshake_mux_0(arch)
    port map(
      index => fork3_outs_2,
      index_valid => fork3_outs_2_valid,
      index_ready => fork3_outs_2_ready,
      ins(0) => buffer0_outs,
      ins(1) => passer9_result,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => passer9_result_valid,
      ins_ready(0) => buffer0_outs_ready,
      ins_ready(1) => passer9_result_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer1 : entity work.handshake_buffer_2(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  mux5 : entity work.handshake_mux_4(arch)
    port map(
      index => fork3_outs_1,
      index_valid => fork3_outs_1_valid,
      index_ready => fork3_outs_1_ready,
      ins(0) => buffer1_outs,
      ins(1) => passer11_result,
      ins_valid(0) => buffer1_outs_valid,
      ins_valid(1) => passer11_result_valid,
      ins_ready(0) => buffer1_outs_ready,
      ins_ready(1) => passer11_result_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.handshake_mux_5(arch)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      ins_valid(0) => control_merge0_outs_valid,
      ins_valid(1) => passer10_result_valid,
      ins_ready(0) => control_merge0_outs_ready,
      ins_ready(1) => passer10_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant6 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant7 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => fork19_outs_3,
      lhs_valid => fork19_outs_3_valid,
      lhs_ready => fork19_outs_3_ready,
      rhs => fork20_outs_1,
      rhs_valid => fork20_outs_1_valid,
      rhs_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  buffer20 : entity work.handshake_buffer_3(arch)
    port map(
      ins => fork19_outs_2,
      ins_valid => fork19_outs_2_valid,
      ins_ready => fork19_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => buffer20_outs,
      lhs_valid => buffer20_outs_valid,
      lhs_ready => buffer20_outs_ready,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  passer7 : entity work.handshake_passer_0(arch)
    port map(
      data => fork4_outs_0,
      data_valid => fork4_outs_0_valid,
      data_ready => fork4_outs_0_ready,
      ctrl => fork2_outs_1,
      ctrl_valid => fork2_outs_1_valid,
      ctrl_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer7_result,
      result_valid => passer7_result_valid,
      result_ready => passer7_result_ready
    );

  fork4 : entity work.handshake_fork_2(arch)
    port map(
      ins => mulf1_result,
      ins_valid => mulf1_result_valid,
      ins_ready => mulf1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  mulf1 : entity work.handshake_mulf_1(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      rhs => constant6_outs,
      rhs_valid => constant6_outs_valid,
      rhs_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready
    );

  buffer19 : entity work.handshake_buffer_4(arch)
    port map(
      ins => fork19_outs_1,
      ins_valid => fork19_outs_1_valid,
      ins_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  subf0 : entity work.handshake_subf_0(arch)
    port map(
      lhs => fork4_outs_1,
      lhs_valid => fork4_outs_1_valid,
      lhs_ready => fork4_outs_1_ready,
      rhs => buffer19_outs,
      rhs_valid => buffer19_outs_valid,
      rhs_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      result => subf0_result,
      result_valid => subf0_result_valid,
      result_ready => subf0_result_ready
    );

  absf0 : entity work.handshake_absf_0(arch)
    port map(
      ins => subf0_result,
      ins_valid => subf0_result_valid,
      ins_ready => subf0_result_ready,
      clk => clk,
      rst => rst,
      outs => absf0_outs,
      outs_valid => absf0_outs_valid,
      outs_ready => absf0_outs_ready
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => absf0_outs,
      lhs_valid => absf0_outs_valid,
      lhs_ready => absf0_outs_ready,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  fork6 : entity work.handshake_fork_3(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  buffer11 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork2_outs_2,
      ins_valid => fork2_outs_2_valid,
      ins_ready => fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  passer8 : entity work.handshake_passer_1(arch)
    port map(
      data => not0_outs,
      data_valid => not0_outs_valid,
      data_ready => not0_outs_ready,
      ctrl => buffer11_outs,
      ctrl_valid => buffer11_outs_valid,
      ctrl_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      result => passer8_result,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  spec_v2_repeating_init0 : entity work.handshake_spec_v2_repeating_init_0(arch)
    port map(
      ins => passer8_result,
      ins_valid => passer8_result_valid,
      ins_ready => passer8_result_ready,
      clk => clk,
      rst => rst,
      outs => spec_v2_repeating_init0_outs,
      outs_valid => spec_v2_repeating_init0_outs_valid,
      outs_ready => spec_v2_repeating_init0_outs_ready
    );

  buffer9 : entity work.handshake_buffer_6(arch)
    port map(
      ins => spec_v2_repeating_init0_outs,
      ins_valid => spec_v2_repeating_init0_outs_valid,
      ins_ready => spec_v2_repeating_init0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer10 : entity work.handshake_buffer_7(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork2 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs(3) => fork2_outs_3,
      outs(4) => fork2_outs_4,
      outs(5) => fork2_outs_5,
      outs(6) => fork2_outs_6,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_valid(5) => fork2_outs_5_valid,
      outs_valid(6) => fork2_outs_6_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready,
      outs_ready(5) => fork2_outs_5_ready,
      outs_ready(6) => fork2_outs_6_ready
    );

  buffer13 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork2_outs_6,
      ins_valid => fork2_outs_6_valid,
      ins_ready => fork2_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  init4 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => init4_outs,
      outs_valid => init4_outs_valid,
      outs_ready => init4_outs_ready
    );

  fork3 : entity work.handshake_fork_5(arch)
    port map(
      ins => init4_outs,
      ins_valid => init4_outs_valid,
      ins_ready => init4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs(2) => fork3_outs_2,
      outs(3) => fork3_outs_3,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_valid(2) => fork3_outs_2_valid,
      outs_valid(3) => fork3_outs_3_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready,
      outs_ready(2) => fork3_outs_2_ready,
      outs_ready(3) => fork3_outs_3_ready
    );

  buffer12 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork2_outs_5,
      ins_valid => fork2_outs_5_valid,
      ins_ready => fork2_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => buffer12_outs,
      rhs_valid => buffer12_outs_valid,
      rhs_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork16 : entity work.handshake_fork_1(arch)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs(2) => fork16_outs_2,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_valid(2) => fork16_outs_2_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready,
      outs_ready(2) => fork16_outs_2_ready
    );

  buffer18 : entity work.handshake_buffer_8(arch)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  passer0 : entity work.handshake_passer_2(arch)
    port map(
      data => buffer18_outs,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      ctrl => fork16_outs_2,
      ctrl_valid => fork16_outs_2_valid,
      ctrl_ready => fork16_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer0_result,
      result_valid => passer0_result_valid,
      result_ready => passer0_result_ready
    );

  buffer15 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork17_outs_1,
      ins_valid => fork17_outs_1_valid,
      ins_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  passer1 : entity work.handshake_passer_3(arch)
    port map(
      data => buffer15_outs,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      ctrl => fork16_outs_1,
      ctrl_valid => fork16_outs_1_valid,
      ctrl_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer1_result,
      result_valid => passer1_result_valid,
      result_ready => passer1_result_ready
    );

  passer9 : entity work.handshake_passer_3(arch)
    port map(
      data => fork17_outs_0,
      data_valid => fork17_outs_0_valid,
      data_ready => fork17_outs_0_ready,
      ctrl => fork2_outs_4,
      ctrl_valid => fork2_outs_4_valid,
      ctrl_ready => fork2_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer9_result,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  buffer4 : entity work.handshake_buffer_10(arch)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.handshake_buffer_1(arch)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork17 : entity work.handshake_fork_6(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  buffer17 : entity work.handshake_buffer_11(arch)
    port map(
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  passer3 : entity work.handshake_passer_4(arch)
    port map(
      data_valid => buffer17_outs_valid,
      data_ready => buffer17_outs_ready,
      ctrl => fork16_outs_0,
      ctrl_valid => fork16_outs_0_valid,
      ctrl_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer3_result_valid,
      result_ready => passer3_result_ready
    );

  passer10 : entity work.handshake_passer_4(arch)
    port map(
      data_valid => fork18_outs_1_valid,
      data_ready => fork18_outs_1_ready,
      ctrl => fork2_outs_3,
      ctrl_valid => fork2_outs_3_valid,
      ctrl_ready => fork2_outs_3_ready,
      clk => clk,
      rst => rst,
      result_valid => passer10_result_valid,
      result_ready => passer10_result_ready
    );

  buffer7 : entity work.handshake_buffer_12(arch)
    port map(
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer8 : entity work.handshake_buffer_13(arch)
    port map(
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork18 : entity work.handshake_fork_7(arch)
    port map(
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  buffer3 : entity work.handshake_buffer_14(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork19 : entity work.handshake_fork_8(arch)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs(2) => fork19_outs_2,
      outs(3) => fork19_outs_3,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_valid(2) => fork19_outs_2_valid,
      outs_valid(3) => fork19_outs_3_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready,
      outs_ready(2) => fork19_outs_2_ready,
      outs_ready(3) => fork19_outs_3_ready
    );

  buffer21 : entity work.handshake_buffer_15(arch)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  passer11 : entity work.handshake_passer_5(arch)
    port map(
      data => buffer21_outs,
      data_valid => buffer21_outs_valid,
      data_ready => buffer21_outs_ready,
      ctrl => fork2_outs_0,
      ctrl_valid => fork2_outs_0_valid,
      ctrl_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer11_result,
      result_valid => passer11_result_valid,
      result_ready => passer11_result_ready
    );

  buffer6 : entity work.handshake_buffer_16(arch)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  fork20 : entity work.handshake_fork_9(arch)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  buffer14 : entity work.handshake_buffer_10(arch)
    port map(
      ins => passer1_result,
      ins_valid => passer1_result_valid,
      ins_ready => passer1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant8 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  fork7 : entity work.handshake_fork_10(arch)
    port map(
      ins => constant8_outs,
      ins_valid => constant8_outs_valid,
      ins_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant1 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi5 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant2 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi6 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  addf1 : entity work.handshake_addf_1(arch)
    port map(
      lhs => passer0_result,
      lhs_valid => passer0_result_valid,
      lhs_ready => passer0_result_ready,
      rhs => fork7_outs_1,
      rhs_valid => fork7_outs_1_valid,
      rhs_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready
    );

  fork8 : entity work.handshake_fork_11(arch)
    port map(
      ins => addf1_result,
      ins_valid => addf1_result_valid,
      ins_ready => addf1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  buffer22 : entity work.handshake_buffer_17(arch)
    port map(
      ins => fork7_outs_0,
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  divf0 : entity work.handshake_divf_0(arch)
    port map(
      lhs => buffer22_outs,
      lhs_valid => buffer22_outs_valid,
      lhs_ready => buffer22_outs_ready,
      rhs => fork8_outs_1,
      rhs_valid => fork8_outs_1_valid,
      rhs_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      result => divf0_result,
      result_valid => divf0_result_valid,
      result_ready => divf0_result_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => extsi4_outs,
      lhs_valid => extsi4_outs_valid,
      lhs_ready => extsi4_outs_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  fork9 : entity work.handshake_fork_12(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork9_outs_1,
      lhs_valid => fork9_outs_1_valid,
      lhs_ready => fork9_outs_1_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork10 : entity work.handshake_fork_5(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs(2) => fork10_outs_2,
      outs(3) => fork10_outs_3,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_valid(2) => fork10_outs_2_valid,
      outs_valid(3) => fork10_outs_3_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready,
      outs_ready(2) => fork10_outs_2_ready,
      outs_ready(3) => fork10_outs_3_ready
    );

  cond_br7 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork10_outs_0,
      condition_valid => fork10_outs_0_valid,
      condition_ready => fork10_outs_0_ready,
      data => trunci0_outs,
      data_valid => trunci0_outs_valid,
      data_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink4 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br8 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork10_outs_1,
      condition_valid => fork10_outs_1_valid,
      condition_ready => fork10_outs_1_ready,
      data => divf0_result,
      data_valid => divf0_result_valid,
      data_ready => divf0_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink5 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer23 : entity work.handshake_buffer_18(arch)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  cond_br9 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork10_outs_2,
      condition_valid => fork10_outs_2_valid,
      condition_ready => fork10_outs_2_ready,
      data => buffer23_outs,
      data_valid => buffer23_outs_valid,
      data_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  buffer16 : entity work.handshake_buffer_12(arch)
    port map(
      ins_valid => passer3_result_valid,
      ins_ready => passer3_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  cond_br10 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork10_outs_3,
      condition_valid => fork10_outs_3_valid,
      condition_ready => fork10_outs_3_ready,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink6 : entity work.handshake_sink_2(arch)
    port map(
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

end architecture;
