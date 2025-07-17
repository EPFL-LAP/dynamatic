library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity iterative_sqrt is
  port (
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of iterative_sqrt is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal cond_br39_trueOut : std_logic_vector(31 downto 0);
  signal cond_br39_trueOut_valid : std_logic;
  signal cond_br39_trueOut_ready : std_logic;
  signal cond_br39_falseOut : std_logic_vector(31 downto 0);
  signal cond_br39_falseOut_valid : std_logic;
  signal cond_br39_falseOut_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal cond_br40_trueOut : std_logic_vector(31 downto 0);
  signal cond_br40_trueOut_valid : std_logic;
  signal cond_br40_trueOut_ready : std_logic;
  signal cond_br40_falseOut : std_logic_vector(31 downto 0);
  signal cond_br40_falseOut_valid : std_logic;
  signal cond_br40_falseOut_ready : std_logic;
  signal cond_br41_trueOut : std_logic_vector(0 downto 0);
  signal cond_br41_trueOut_valid : std_logic;
  signal cond_br41_trueOut_ready : std_logic;
  signal cond_br41_falseOut : std_logic_vector(0 downto 0);
  signal cond_br41_falseOut_valid : std_logic;
  signal cond_br41_falseOut_ready : std_logic;
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal merge0_outs : std_logic_vector(0 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(0 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(0 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(0 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork4_outs_3 : std_logic_vector(0 downto 0);
  signal fork4_outs_3_valid : std_logic;
  signal fork4_outs_3_ready : std_logic;
  signal fork4_outs_4 : std_logic_vector(0 downto 0);
  signal fork4_outs_4_valid : std_logic;
  signal fork4_outs_4_ready : std_logic;
  signal fork4_outs_5 : std_logic_vector(0 downto 0);
  signal fork4_outs_5_valid : std_logic;
  signal fork4_outs_5_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(31 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(0 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(0 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal mux9_outs : std_logic_vector(31 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal mux10_outs : std_logic_vector(0 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(31 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(31 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal fork8_outs_4 : std_logic_vector(31 downto 0);
  signal fork8_outs_4_valid : std_logic;
  signal fork8_outs_4_ready : std_logic;
  signal mux12_outs : std_logic_vector(31 downto 0);
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_2 : std_logic_vector(31 downto 0);
  signal fork9_outs_2_valid : std_logic;
  signal fork9_outs_2_ready : std_logic;
  signal fork9_outs_3 : std_logic_vector(31 downto 0);
  signal fork9_outs_3_valid : std_logic;
  signal fork9_outs_3_ready : std_logic;
  signal fork9_outs_4 : std_logic_vector(31 downto 0);
  signal fork9_outs_4_valid : std_logic;
  signal fork9_outs_4_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(0 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(0 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(0 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(1 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal fork11_outs_3 : std_logic_vector(0 downto 0);
  signal fork11_outs_3_valid : std_logic;
  signal fork11_outs_3_ready : std_logic;
  signal fork11_outs_4 : std_logic_vector(0 downto 0);
  signal fork11_outs_4_valid : std_logic;
  signal fork11_outs_4_ready : std_logic;
  signal fork11_outs_5 : std_logic_vector(0 downto 0);
  signal fork11_outs_5_valid : std_logic;
  signal fork11_outs_5_ready : std_logic;
  signal fork11_outs_6 : std_logic_vector(0 downto 0);
  signal fork11_outs_6_valid : std_logic;
  signal fork11_outs_6_ready : std_logic;
  signal fork11_outs_7 : std_logic_vector(0 downto 0);
  signal fork11_outs_7_valid : std_logic;
  signal fork11_outs_7_ready : std_logic;
  signal fork11_outs_8 : std_logic_vector(0 downto 0);
  signal fork11_outs_8_valid : std_logic;
  signal fork11_outs_8_ready : std_logic;
  signal fork11_outs_9 : std_logic_vector(0 downto 0);
  signal fork11_outs_9_valid : std_logic;
  signal fork11_outs_9_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal shrsi0_result : std_logic_vector(31 downto 0);
  signal shrsi0_result_valid : std_logic;
  signal shrsi0_result_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(31 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(31 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal xori0_result : std_logic_vector(0 downto 0);
  signal xori0_result_valid : std_logic;
  signal xori0_result_ready : std_logic;
  signal andi2_result : std_logic_vector(0 downto 0);
  signal andi2_result_valid : std_logic;
  signal andi2_result_ready : std_logic;
  signal andi3_result : std_logic_vector(0 downto 0);
  signal andi3_result_valid : std_logic;
  signal andi3_result_ready : std_logic;
  signal ori0_result : std_logic_vector(0 downto 0);
  signal ori0_result_valid : std_logic;
  signal ori0_result_ready : std_logic;
  signal xori1_result : std_logic_vector(0 downto 0);
  signal xori1_result_valid : std_logic;
  signal xori1_result_ready : std_logic;
  signal andi4_result : std_logic_vector(0 downto 0);
  signal andi4_result_valid : std_logic;
  signal andi4_result_ready : std_logic;
  signal andi5_result : std_logic_vector(0 downto 0);
  signal andi5_result_valid : std_logic;
  signal andi5_result_ready : std_logic;
  signal ori1_result : std_logic_vector(0 downto 0);
  signal ori1_result_valid : std_logic;
  signal ori1_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal cond_br42_trueOut : std_logic_vector(31 downto 0);
  signal cond_br42_trueOut_valid : std_logic;
  signal cond_br42_trueOut_ready : std_logic;
  signal cond_br42_falseOut : std_logic_vector(31 downto 0);
  signal cond_br42_falseOut_valid : std_logic;
  signal cond_br42_falseOut_ready : std_logic;
  signal cond_br43_trueOut : std_logic_vector(31 downto 0);
  signal cond_br43_trueOut_valid : std_logic;
  signal cond_br43_trueOut_ready : std_logic;
  signal cond_br43_falseOut : std_logic_vector(31 downto 0);
  signal cond_br43_falseOut_valid : std_logic;
  signal cond_br43_falseOut_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork13_outs_2 : std_logic_vector(31 downto 0);
  signal fork13_outs_2_valid : std_logic;
  signal fork13_outs_2_ready : std_logic;
  signal cond_br44_trueOut : std_logic_vector(31 downto 0);
  signal cond_br44_trueOut_valid : std_logic;
  signal cond_br44_trueOut_ready : std_logic;
  signal cond_br44_falseOut : std_logic_vector(31 downto 0);
  signal cond_br44_falseOut_valid : std_logic;
  signal cond_br44_falseOut_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(31 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(31 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(31 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(1 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal shrsi1_result : std_logic_vector(31 downto 0);
  signal shrsi1_result_valid : std_logic;
  signal shrsi1_result_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork15_outs_2 : std_logic_vector(31 downto 0);
  signal fork15_outs_2_valid : std_logic;
  signal fork15_outs_2_ready : std_logic;
  signal fork15_outs_3 : std_logic_vector(31 downto 0);
  signal fork15_outs_3_valid : std_logic;
  signal fork15_outs_3_ready : std_logic;
  signal muli1_result : std_logic_vector(31 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(31 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(31 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal fork16_outs_2 : std_logic_vector(31 downto 0);
  signal fork16_outs_2_valid : std_logic;
  signal fork16_outs_2_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal cmpi4_result : std_logic_vector(0 downto 0);
  signal cmpi4_result_valid : std_logic;
  signal cmpi4_result_ready : std_logic;
  signal cmpi5_result : std_logic_vector(0 downto 0);
  signal cmpi5_result_valid : std_logic;
  signal cmpi5_result_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(0 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(0 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_2 : std_logic_vector(0 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal andi6_result : std_logic_vector(0 downto 0);
  signal andi6_result_valid : std_logic;
  signal andi6_result_ready : std_logic;
  signal select0_result : std_logic_vector(31 downto 0);
  signal select0_result_valid : std_logic;
  signal select0_result_ready : std_logic;
  signal ori2_result : std_logic_vector(0 downto 0);
  signal ori2_result_valid : std_logic;
  signal ori2_result_ready : std_logic;
  signal cond_br45_trueOut : std_logic_vector(31 downto 0);
  signal cond_br45_trueOut_valid : std_logic;
  signal cond_br45_trueOut_ready : std_logic;
  signal cond_br45_falseOut : std_logic_vector(31 downto 0);
  signal cond_br45_falseOut_valid : std_logic;
  signal cond_br45_falseOut_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(31 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(31 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(31 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal fork18_outs_3 : std_logic_vector(31 downto 0);
  signal fork18_outs_3_valid : std_logic;
  signal fork18_outs_3_ready : std_logic;
  signal cmpi6_result : std_logic_vector(0 downto 0);
  signal cmpi6_result_valid : std_logic;
  signal cmpi6_result_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(0 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(0 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(0 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal cond_br46_trueOut : std_logic_vector(31 downto 0);
  signal cond_br46_trueOut_valid : std_logic;
  signal cond_br46_trueOut_ready : std_logic;
  signal cond_br46_falseOut : std_logic_vector(31 downto 0);
  signal cond_br46_falseOut_valid : std_logic;
  signal cond_br46_falseOut_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant11_outs : std_logic_vector(1 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(31 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal select1_result : std_logic_vector(31 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;

begin

  out0 <= select1_result;
  out0_valid <= select1_result_valid;
  select1_result_ready <= out0_ready;
  end_valid <= fork0_outs_0_valid;
  fork0_outs_0_ready <= end_ready;

  fork0 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => n,
      ins_valid => n_valid,
      ins_ready => n_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant2 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi0 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi0_outs,
      outs_valid => extsi0_outs_valid,
      outs_ready => extsi0_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant3 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  buffer8 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  cond_br39 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_6,
      condition_valid => fork11_outs_6_valid,
      condition_ready => fork11_outs_6_ready,
      data => buffer9_outs,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br39_trueOut,
      trueOut_valid => cond_br39_trueOut_valid,
      trueOut_ready => cond_br39_trueOut_ready,
      falseOut => cond_br39_falseOut,
      falseOut_valid => cond_br39_falseOut_valid,
      falseOut_ready => cond_br39_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br39_falseOut,
      ins_valid => cond_br39_falseOut_valid,
      ins_ready => cond_br39_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer4 : entity work.oehb(arch) generic map(32)
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

  buffer5 : entity work.tehb(arch) generic map(32)
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

  cond_br40 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_7,
      condition_valid => fork11_outs_7_valid,
      condition_ready => fork11_outs_7_ready,
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br40_trueOut,
      trueOut_valid => cond_br40_trueOut_valid,
      trueOut_ready => cond_br40_trueOut_ready,
      falseOut => cond_br40_falseOut,
      falseOut_valid => cond_br40_falseOut_valid,
      falseOut_ready => cond_br40_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br40_falseOut,
      ins_valid => cond_br40_falseOut_valid,
      ins_ready => cond_br40_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br41 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork11_outs_8,
      condition_valid => fork11_outs_8_valid,
      condition_ready => fork11_outs_8_ready,
      data => ori1_result,
      data_valid => ori1_result_valid,
      data_ready => ori1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br41_trueOut,
      trueOut_valid => cond_br41_trueOut_valid,
      trueOut_ready => cond_br41_trueOut_ready,
      falseOut => cond_br41_falseOut,
      falseOut_valid => cond_br41_falseOut_valid,
      falseOut_ready => cond_br41_falseOut_ready
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi1 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  merge0 : entity work.merge(arch) generic map(2, 1)
    port map(
      ins(0) => fork3_outs_0,
      ins(1) => fork11_outs_9,
      ins_valid(0) => fork3_outs_0_valid,
      ins_valid(1) => fork11_outs_9_valid,
      ins_ready(0) => fork3_outs_0_ready,
      ins_ready(1) => fork11_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  buffer0 : entity work.oehb(arch) generic map(1)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  buffer1 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs(2) => fork4_outs_2,
      outs(3) => fork4_outs_3,
      outs(4) => fork4_outs_4,
      outs(5) => fork4_outs_5,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_valid(3) => fork4_outs_3_valid,
      outs_valid(4) => fork4_outs_4_valid,
      outs_valid(5) => fork4_outs_5_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready,
      outs_ready(3) => fork4_outs_3_ready,
      outs_ready(4) => fork4_outs_4_ready,
      outs_ready(5) => fork4_outs_5_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_5,
      index_valid => fork4_outs_5_valid,
      index_ready => fork4_outs_5_ready,
      ins(0) => fork1_outs_1,
      ins(1) => fork18_outs_1,
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => fork18_outs_1_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer2 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  buffer6 : entity work.oehb(arch) generic map(32)
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

  buffer7 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork17_outs_1,
      index_valid => fork17_outs_1_valid,
      index_ready => fork17_outs_1_ready,
      ins(0) => buffer7_outs,
      ins(1) => fork8_outs_3,
      ins_valid(0) => buffer7_outs_valid,
      ins_valid(1) => fork8_outs_3_valid,
      ins_ready(0) => buffer7_outs_ready,
      ins_ready(1) => fork8_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork19_outs_1,
      index_valid => fork19_outs_1_valid,
      index_ready => fork19_outs_1_ready,
      ins(0) => fork8_outs_4,
      ins(1) => addi2_result,
      ins_valid(0) => fork8_outs_4_valid,
      ins_valid(1) => addi2_result_valid,
      ins_ready(0) => fork8_outs_4_ready,
      ins_ready(1) => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer10 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork17_outs_2,
      index_valid => fork17_outs_2_valid,
      index_ready => fork17_outs_2_ready,
      ins(0) => buffer11_outs,
      ins(1) => fork9_outs_3,
      ins_valid(0) => buffer11_outs_valid,
      ins_valid(1) => fork9_outs_3_valid,
      ins_ready(0) => buffer11_outs_ready,
      ins_ready(1) => fork9_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork19_outs_2,
      index_valid => fork19_outs_2_valid,
      index_ready => fork19_outs_2_ready,
      ins(0) => addi3_result,
      ins(1) => fork9_outs_4,
      ins_valid(0) => addi3_result_valid,
      ins_valid(1) => fork9_outs_4_valid,
      ins_ready(0) => addi3_result_ready,
      ins_ready(1) => fork9_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork4_outs_4,
      index_valid => fork4_outs_4_valid,
      index_ready => fork4_outs_4_ready,
      ins(0) => fork2_outs_1,
      ins(1) => ori2_result,
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => ori2_result_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => ori2_result_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  buffer12 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_3,
      index_valid => fork4_outs_3_valid,
      index_ready => fork4_outs_3_ready,
      ins(0) => extsi1_outs,
      ins(1) => select0_result,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => select0_result_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => select0_result_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  mux10 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork4_outs_2,
      index_valid => fork4_outs_2_valid,
      index_ready => fork4_outs_2_ready,
      ins(0) => fork2_outs_0,
      ins(1) => cond_br41_trueOut,
      ins_valid(0) => fork2_outs_0_valid,
      ins_valid(1) => cond_br41_trueOut_valid,
      ins_ready(0) => fork2_outs_0_ready,
      ins_ready(1) => cond_br41_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  buffer16 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux10_outs,
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer17 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => extsi0_outs,
      ins(1) => cond_br40_trueOut,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => cond_br40_trueOut_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => cond_br40_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  buffer18 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux11_outs,
      ins_valid => mux11_outs_valid,
      ins_ready => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  buffer19 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs(4) => fork8_outs_4,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_valid(4) => fork8_outs_4_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready,
      outs_ready(4) => fork8_outs_4_ready
    );

  mux12 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br39_trueOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br39_trueOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => cond_br39_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux12_outs,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  buffer20 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux12_outs,
      ins_valid => mux12_outs_valid,
      ins_ready => mux12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  buffer21 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs(2) => fork9_outs_2,
      outs(3) => fork9_outs_3,
      outs(4) => fork9_outs_4,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_valid(2) => fork9_outs_2_valid,
      outs_valid(3) => fork9_outs_3_valid,
      outs_valid(4) => fork9_outs_4_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready,
      outs_ready(2) => fork9_outs_2_ready,
      outs_ready(3) => fork9_outs_3_ready,
      outs_ready(4) => fork9_outs_4_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant9 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork8_outs_2,
      lhs_valid => fork8_outs_2_valid,
      lhs_ready => fork8_outs_2_ready,
      rhs => fork9_outs_2,
      rhs_valid => fork9_outs_2_valid,
      rhs_ready => fork9_outs_2_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  andi0 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      rhs => fork6_outs_2,
      rhs_valid => fork6_outs_2_valid,
      rhs_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(10, 1)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs(2) => fork11_outs_2,
      outs(3) => fork11_outs_3,
      outs(4) => fork11_outs_4,
      outs(5) => fork11_outs_5,
      outs(6) => fork11_outs_6,
      outs(7) => fork11_outs_7,
      outs(8) => fork11_outs_8,
      outs(9) => fork11_outs_9,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_valid(3) => fork11_outs_3_valid,
      outs_valid(4) => fork11_outs_4_valid,
      outs_valid(5) => fork11_outs_5_valid,
      outs_valid(6) => fork11_outs_6_valid,
      outs_valid(7) => fork11_outs_7_valid,
      outs_valid(8) => fork11_outs_8_valid,
      outs_valid(9) => fork11_outs_9_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready,
      outs_ready(3) => fork11_outs_3_ready,
      outs_ready(4) => fork11_outs_4_ready,
      outs_ready(5) => fork11_outs_5_ready,
      outs_ready(6) => fork11_outs_6_ready,
      outs_ready(7) => fork11_outs_7_ready,
      outs_ready(8) => fork11_outs_8_ready,
      outs_ready(9) => fork11_outs_9_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork8_outs_1,
      lhs_valid => fork8_outs_1_valid,
      lhs_ready => fork8_outs_1_ready,
      rhs => fork9_outs_1,
      rhs_valid => fork9_outs_1_valid,
      rhs_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  shrsi0 : entity work.shrsi(arch) generic map(32)
    port map(
      lhs => addi0_result,
      lhs_valid => addi0_result_valid,
      lhs_ready => addi0_result_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => shrsi0_result,
      result_valid => shrsi0_result_valid,
      result_ready => shrsi0_result_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => shrsi0_result,
      ins_valid => shrsi0_result_valid,
      ins_ready => shrsi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => fork12_outs_0,
      lhs_valid => fork12_outs_0_valid,
      lhs_ready => fork12_outs_0_ready,
      rhs => fork12_outs_1,
      rhs_valid => fork12_outs_1_valid,
      rhs_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => muli0_result,
      lhs_valid => muli0_result_valid,
      lhs_ready => muli0_result_ready,
      rhs => fork5_outs_1,
      rhs_valid => fork5_outs_1_valid,
      rhs_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  andi1 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi1_result,
      lhs_valid => cmpi1_result_valid,
      lhs_ready => cmpi1_result_ready,
      rhs => fork7_outs_2,
      rhs_valid => fork7_outs_2_valid,
      rhs_ready => fork7_outs_2_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  xori0 : entity work.xori(arch) generic map(1)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => fork10_outs_1,
      rhs_valid => fork10_outs_1_valid,
      rhs_ready => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      result => xori0_result,
      result_valid => xori0_result_valid,
      result_ready => xori0_result_ready
    );

  andi2 : entity work.andi(arch) generic map(1)
    port map(
      lhs => fork6_outs_0,
      lhs_valid => fork6_outs_0_valid,
      lhs_ready => fork6_outs_0_ready,
      rhs => andi1_result,
      rhs_valid => andi1_result_valid,
      rhs_ready => andi1_result_ready,
      clk => clk,
      rst => rst,
      result => andi2_result,
      result_valid => andi2_result_valid,
      result_ready => andi2_result_ready
    );

  andi3 : entity work.andi(arch) generic map(1)
    port map(
      lhs => xori0_result,
      lhs_valid => xori0_result_valid,
      lhs_ready => xori0_result_ready,
      rhs => fork7_outs_1,
      rhs_valid => fork7_outs_1_valid,
      rhs_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      result => andi3_result,
      result_valid => andi3_result_valid,
      result_ready => andi3_result_ready
    );

  ori0 : entity work.ori(arch) generic map(1)
    port map(
      lhs => andi2_result,
      lhs_valid => andi2_result_valid,
      lhs_ready => andi2_result_ready,
      rhs => andi3_result,
      rhs_valid => andi3_result_valid,
      rhs_ready => andi3_result_ready,
      clk => clk,
      rst => rst,
      result => ori0_result,
      result_valid => ori0_result_valid,
      result_ready => ori0_result_ready
    );

  xori1 : entity work.xori(arch) generic map(1)
    port map(
      lhs => fork11_outs_5,
      lhs_valid => fork11_outs_5_valid,
      lhs_ready => fork11_outs_5_ready,
      rhs => fork10_outs_0,
      rhs_valid => fork10_outs_0_valid,
      rhs_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      result => xori1_result,
      result_valid => xori1_result_valid,
      result_ready => xori1_result_ready
    );

  andi4 : entity work.andi(arch) generic map(1)
    port map(
      lhs => fork11_outs_4,
      lhs_valid => fork11_outs_4_valid,
      lhs_ready => fork11_outs_4_ready,
      rhs => ori0_result,
      rhs_valid => ori0_result_valid,
      rhs_ready => ori0_result_ready,
      clk => clk,
      rst => rst,
      result => andi4_result,
      result_valid => andi4_result_valid,
      result_ready => andi4_result_ready
    );

  andi5 : entity work.andi(arch) generic map(1)
    port map(
      lhs => xori1_result,
      lhs_valid => xori1_result_valid,
      lhs_ready => xori1_result_ready,
      rhs => fork7_outs_0,
      rhs_valid => fork7_outs_0_valid,
      rhs_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi5_result,
      result_valid => andi5_result_valid,
      result_ready => andi5_result_ready
    );

  ori1 : entity work.ori(arch) generic map(1)
    port map(
      lhs => andi4_result,
      lhs_valid => andi4_result_valid,
      lhs_ready => andi4_result_ready,
      rhs => andi5_result,
      rhs_valid => andi5_result_valid,
      rhs_ready => andi5_result_ready,
      clk => clk,
      rst => rst,
      result => ori1_result,
      result_valid => ori1_result_valid,
      result_ready => ori1_result_ready
    );

  buffer14 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  buffer15 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  cond_br42 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_3,
      condition_valid => fork11_outs_3_valid,
      condition_ready => fork11_outs_3_ready,
      data => buffer15_outs,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br42_trueOut,
      trueOut_valid => cond_br42_trueOut_valid,
      trueOut_ready => cond_br42_trueOut_ready,
      falseOut => cond_br42_falseOut,
      falseOut_valid => cond_br42_falseOut_valid,
      falseOut_ready => cond_br42_falseOut_ready
    );

  cond_br43 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_2,
      condition_valid => fork11_outs_2_valid,
      condition_ready => fork11_outs_2_ready,
      data => fork8_outs_0,
      data_valid => fork8_outs_0_valid,
      data_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br43_trueOut,
      trueOut_valid => cond_br43_trueOut_valid,
      trueOut_ready => cond_br43_trueOut_ready,
      falseOut => cond_br43_falseOut,
      falseOut_valid => cond_br43_falseOut_valid,
      falseOut_ready => cond_br43_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br43_falseOut,
      ins_valid => cond_br43_falseOut_valid,
      ins_ready => cond_br43_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork13 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => cond_br43_trueOut,
      ins_valid => cond_br43_trueOut_valid,
      ins_ready => cond_br43_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready
    );

  cond_br44 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_1,
      condition_valid => fork11_outs_1_valid,
      condition_ready => fork11_outs_1_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br44_trueOut,
      trueOut_valid => cond_br44_trueOut_valid,
      trueOut_ready => cond_br44_trueOut_ready,
      falseOut => cond_br44_falseOut,
      falseOut_valid => cond_br44_falseOut_valid,
      falseOut_ready => cond_br44_falseOut_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => cond_br44_trueOut,
      ins_valid => cond_br44_trueOut_valid,
      ins_ready => cond_br44_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant10 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant10_outs,
      ins_valid => constant10_outs_valid,
      ins_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork13_outs_2,
      lhs_valid => fork13_outs_2_valid,
      lhs_ready => fork13_outs_2_ready,
      rhs => fork14_outs_2,
      rhs_valid => fork14_outs_2_valid,
      rhs_ready => fork14_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  shrsi1 : entity work.shrsi(arch) generic map(32)
    port map(
      lhs => addi1_result,
      lhs_valid => addi1_result_valid,
      lhs_ready => addi1_result_ready,
      rhs => extsi3_outs,
      rhs_valid => extsi3_outs_valid,
      rhs_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      result => shrsi1_result,
      result_valid => shrsi1_result_valid,
      result_ready => shrsi1_result_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => shrsi1_result,
      ins_valid => shrsi1_result_valid,
      ins_ready => shrsi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs(2) => fork15_outs_2,
      outs(3) => fork15_outs_3,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_valid(2) => fork15_outs_2_valid,
      outs_valid(3) => fork15_outs_3_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready,
      outs_ready(2) => fork15_outs_2_ready,
      outs_ready(3) => fork15_outs_3_ready
    );

  muli1 : entity work.muli(arch) generic map(32)
    port map(
      lhs => fork15_outs_2,
      lhs_valid => fork15_outs_2_valid,
      lhs_ready => fork15_outs_2_ready,
      rhs => fork15_outs_3,
      rhs_valid => fork15_outs_3_valid,
      rhs_ready => fork15_outs_3_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => muli1_result,
      ins_valid => muli1_result_valid,
      ins_ready => muli1_result_ready,
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

  cmpi2 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork16_outs_2,
      lhs_valid => fork16_outs_2_valid,
      lhs_ready => fork16_outs_2_ready,
      rhs => fork18_outs_2,
      rhs_valid => fork18_outs_2_valid,
      rhs_ready => fork18_outs_2_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  cmpi3 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork13_outs_1,
      lhs_valid => fork13_outs_1_valid,
      lhs_ready => fork13_outs_1_ready,
      rhs => fork14_outs_1,
      rhs_valid => fork14_outs_1_valid,
      rhs_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  cmpi4 : entity work.handshake_cmpi_2(arch) generic map(32)
    port map(
      lhs => fork13_outs_0,
      lhs_valid => fork13_outs_0_valid,
      lhs_ready => fork13_outs_0_ready,
      rhs => fork14_outs_0,
      rhs_valid => fork14_outs_0_valid,
      rhs_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi4_result,
      result_valid => cmpi4_result_valid,
      result_ready => cmpi4_result_ready
    );

  cmpi5 : entity work.handshake_cmpi_3(arch) generic map(32)
    port map(
      lhs => fork16_outs_1,
      lhs_valid => fork16_outs_1_valid,
      lhs_ready => fork16_outs_1_ready,
      rhs => fork18_outs_3,
      rhs_valid => fork18_outs_3_valid,
      rhs_ready => fork18_outs_3_ready,
      clk => clk,
      rst => rst,
      result => cmpi5_result,
      result_valid => cmpi5_result_valid,
      result_ready => cmpi5_result_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => cmpi5_result,
      ins_valid => cmpi5_result_valid,
      ins_ready => cmpi5_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready
    );

  andi6 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi3_result,
      lhs_valid => cmpi3_result_valid,
      lhs_ready => cmpi3_result_ready,
      rhs => cmpi2_result,
      rhs_valid => cmpi2_result_valid,
      rhs_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      result => andi6_result,
      result_valid => andi6_result_valid,
      result_ready => andi6_result_ready
    );

  select0 : entity work.selector(arch) generic map(32)
    port map(
      condition => fork17_outs_0,
      condition_valid => fork17_outs_0_valid,
      condition_ready => fork17_outs_0_ready,
      trueValue => fork15_outs_1,
      trueValue_valid => fork15_outs_1_valid,
      trueValue_ready => fork15_outs_1_ready,
      falseValue => cond_br42_trueOut,
      falseValue_valid => cond_br42_trueOut_valid,
      falseValue_ready => cond_br42_trueOut_ready,
      clk => clk,
      rst => rst,
      result => select0_result,
      result_valid => select0_result_valid,
      result_ready => select0_result_ready
    );

  ori2 : entity work.ori(arch) generic map(1)
    port map(
      lhs => andi6_result,
      lhs_valid => andi6_result_valid,
      lhs_ready => andi6_result_ready,
      rhs => cmpi4_result,
      rhs_valid => cmpi4_result_valid,
      rhs_ready => cmpi4_result_ready,
      clk => clk,
      rst => rst,
      result => ori2_result,
      result_valid => ori2_result_valid,
      result_ready => ori2_result_ready
    );

  cond_br45 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork11_outs_0,
      condition_valid => fork11_outs_0_valid,
      condition_ready => fork11_outs_0_ready,
      data => fork5_outs_0,
      data_valid => fork5_outs_0_valid,
      data_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br45_trueOut,
      trueOut_valid => cond_br45_trueOut_valid,
      trueOut_ready => cond_br45_trueOut_ready,
      falseOut => cond_br45_falseOut,
      falseOut_valid => cond_br45_falseOut_valid,
      falseOut_ready => cond_br45_falseOut_ready
    );

  sink3 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br45_falseOut,
      ins_valid => cond_br45_falseOut_valid,
      ins_ready => cond_br45_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork18 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => cond_br45_trueOut,
      ins_valid => cond_br45_trueOut_valid,
      ins_ready => cond_br45_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs(3) => fork18_outs_3,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_valid(3) => fork18_outs_3_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready,
      outs_ready(3) => fork18_outs_3_ready
    );

  cmpi6 : entity work.handshake_cmpi_4(arch) generic map(32)
    port map(
      lhs => fork16_outs_0,
      lhs_valid => fork16_outs_0_valid,
      lhs_ready => fork16_outs_0_ready,
      rhs => fork18_outs_0,
      rhs_valid => fork18_outs_0_valid,
      rhs_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi6_result,
      result_valid => cmpi6_result_valid,
      result_ready => cmpi6_result_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => cmpi6_result,
      ins_valid => cmpi6_result_valid,
      ins_ready => cmpi6_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs(2) => fork19_outs_2,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_valid(2) => fork19_outs_2_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready,
      outs_ready(2) => fork19_outs_2_ready
    );

  cond_br46 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork19_outs_0,
      condition_valid => fork19_outs_0_valid,
      condition_ready => fork19_outs_0_ready,
      data => fork15_outs_0,
      data_valid => fork15_outs_0_valid,
      data_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br46_trueOut,
      trueOut_valid => cond_br46_trueOut_valid,
      trueOut_ready => cond_br46_trueOut_ready,
      falseOut => cond_br46_falseOut,
      falseOut_valid => cond_br46_falseOut_valid,
      falseOut_ready => cond_br46_falseOut_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant11 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant11_outs,
      ins_valid => constant11_outs_valid,
      ins_ready => constant11_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br46_trueOut,
      lhs_valid => cond_br46_trueOut_valid,
      lhs_ready => cond_br46_trueOut_ready,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  source10 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  constant8 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source10_outs_valid,
      ctrl_ready => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br46_falseOut,
      lhs_valid => cond_br46_falseOut_valid,
      lhs_ready => cond_br46_falseOut_ready,
      rhs => constant8_outs,
      rhs_valid => constant8_outs_valid,
      rhs_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  select1 : entity work.selector(arch) generic map(32)
    port map(
      condition => cond_br41_falseOut,
      condition_valid => cond_br41_falseOut_valid,
      condition_ready => cond_br41_falseOut_ready,
      trueValue => cond_br44_falseOut,
      trueValue_valid => cond_br44_falseOut_valid,
      trueValue_ready => cond_br44_falseOut_ready,
      falseValue => cond_br42_falseOut,
      falseValue_valid => cond_br42_falseOut_valid,
      falseValue_ready => cond_br42_falseOut_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

end architecture;
