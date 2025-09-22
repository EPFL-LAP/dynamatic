library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bisection is
  port (
    a : in std_logic_vector(31 downto 0);
    a_valid : in std_logic;
    b : in std_logic_vector(31 downto 0);
    b_valid : in std_logic;
    tol : in std_logic_vector(31 downto 0);
    tol_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    a_ready : out std_logic;
    b_ready : out std_logic;
    tol_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of bisection is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(31 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(31 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal extsi3_outs : std_logic_vector(7 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(7 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
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
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork47_outs_0 : std_logic_vector(31 downto 0);
  signal fork47_outs_0_valid : std_logic;
  signal fork47_outs_0_ready : std_logic;
  signal fork47_outs_1 : std_logic_vector(31 downto 0);
  signal fork47_outs_1_valid : std_logic;
  signal fork47_outs_1_ready : std_logic;
  signal fork47_outs_2 : std_logic_vector(31 downto 0);
  signal fork47_outs_2_valid : std_logic;
  signal fork47_outs_2_ready : std_logic;
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal fork48_outs_0 : std_logic_vector(31 downto 0);
  signal fork48_outs_0_valid : std_logic;
  signal fork48_outs_0_ready : std_logic;
  signal fork48_outs_1 : std_logic_vector(31 downto 0);
  signal fork48_outs_1_valid : std_logic;
  signal fork48_outs_1_ready : std_logic;
  signal fork48_outs_2 : std_logic_vector(31 downto 0);
  signal fork48_outs_2_valid : std_logic;
  signal fork48_outs_2_ready : std_logic;
  signal buffer2_outs : std_logic_vector(7 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(7 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(31 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(31 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork49_outs_0 : std_logic_vector(31 downto 0);
  signal fork49_outs_0_valid : std_logic;
  signal fork49_outs_0_ready : std_logic;
  signal fork49_outs_1 : std_logic_vector(31 downto 0);
  signal fork49_outs_1_valid : std_logic;
  signal fork49_outs_1_ready : std_logic;
  signal fork49_outs_2 : std_logic_vector(31 downto 0);
  signal fork49_outs_2_valid : std_logic;
  signal fork49_outs_2_ready : std_logic;
  signal fork49_outs_3 : std_logic_vector(31 downto 0);
  signal fork49_outs_3_valid : std_logic;
  signal fork49_outs_3_ready : std_logic;
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork50_outs_0_valid : std_logic;
  signal fork50_outs_0_ready : std_logic;
  signal fork50_outs_1_valid : std_logic;
  signal fork50_outs_1_ready : std_logic;
  signal fork50_outs_2_valid : std_logic;
  signal fork50_outs_2_ready : std_logic;
  signal fork50_outs_3_valid : std_logic;
  signal fork50_outs_3_ready : std_logic;
  signal fork50_outs_4_valid : std_logic;
  signal fork50_outs_4_ready : std_logic;
  signal fork50_outs_5_valid : std_logic;
  signal fork50_outs_5_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(31 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal fork51_outs_0 : std_logic_vector(31 downto 0);
  signal fork51_outs_0_valid : std_logic;
  signal fork51_outs_0_ready : std_logic;
  signal fork51_outs_1 : std_logic_vector(31 downto 0);
  signal fork51_outs_1_valid : std_logic;
  signal fork51_outs_1_ready : std_logic;
  signal fork51_outs_2 : std_logic_vector(31 downto 0);
  signal fork51_outs_2_valid : std_logic;
  signal fork51_outs_2_ready : std_logic;
  signal fork51_outs_3 : std_logic_vector(31 downto 0);
  signal fork51_outs_3_valid : std_logic;
  signal fork51_outs_3_ready : std_logic;
  signal fork51_outs_4 : std_logic_vector(31 downto 0);
  signal fork51_outs_4_valid : std_logic;
  signal fork51_outs_4_ready : std_logic;
  signal fork51_outs_5 : std_logic_vector(31 downto 0);
  signal fork51_outs_5_valid : std_logic;
  signal fork51_outs_5_ready : std_logic;
  signal mulf2_result : std_logic_vector(31 downto 0);
  signal mulf2_result_valid : std_logic;
  signal mulf2_result_ready : std_logic;
  signal addf2_result : std_logic_vector(31 downto 0);
  signal addf2_result_valid : std_logic;
  signal addf2_result_ready : std_logic;
  signal fork52_outs_0 : std_logic_vector(31 downto 0);
  signal fork52_outs_0_valid : std_logic;
  signal fork52_outs_0_ready : std_logic;
  signal fork52_outs_1 : std_logic_vector(31 downto 0);
  signal fork52_outs_1_valid : std_logic;
  signal fork52_outs_1_ready : std_logic;
  signal fork52_outs_2 : std_logic_vector(31 downto 0);
  signal fork52_outs_2_valid : std_logic;
  signal fork52_outs_2_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal absf0_outs : std_logic_vector(31 downto 0);
  signal absf0_outs_valid : std_logic;
  signal absf0_outs_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(0 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(0 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork25_outs_2 : std_logic_vector(0 downto 0);
  signal fork25_outs_2_valid : std_logic;
  signal fork25_outs_2_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal fork7_outs_3 : std_logic_vector(0 downto 0);
  signal fork7_outs_3_valid : std_logic;
  signal fork7_outs_3_ready : std_logic;
  signal fork7_outs_4 : std_logic_vector(0 downto 0);
  signal fork7_outs_4_valid : std_logic;
  signal fork7_outs_4_ready : std_logic;
  signal fork7_outs_5 : std_logic_vector(0 downto 0);
  signal fork7_outs_5_valid : std_logic;
  signal fork7_outs_5_ready : std_logic;
  signal fork7_outs_6 : std_logic_vector(0 downto 0);
  signal fork7_outs_6_valid : std_logic;
  signal fork7_outs_6_ready : std_logic;
  signal fork7_outs_7 : std_logic_vector(0 downto 0);
  signal fork7_outs_7_valid : std_logic;
  signal fork7_outs_7_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(0 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(0 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal andi2_result : std_logic_vector(0 downto 0);
  signal andi2_result_valid : std_logic;
  signal andi2_result_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal passer0_result : std_logic_vector(31 downto 0);
  signal passer0_result_valid : std_logic;
  signal passer0_result_ready : std_logic;
  signal passer2_result_valid : std_logic;
  signal passer2_result_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal subf0_result : std_logic_vector(31 downto 0);
  signal subf0_result_valid : std_logic;
  signal subf0_result_ready : std_logic;
  signal mulf3_result : std_logic_vector(31 downto 0);
  signal mulf3_result_valid : std_logic;
  signal mulf3_result_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal cmpf1_result : std_logic_vector(0 downto 0);
  signal cmpf1_result_valid : std_logic;
  signal cmpf1_result_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(0 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(0 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal not1_outs : std_logic_vector(0 downto 0);
  signal not1_outs_valid : std_logic;
  signal not1_outs_ready : std_logic;
  signal andi3_result : std_logic_vector(0 downto 0);
  signal andi3_result_valid : std_logic;
  signal andi3_result_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(0 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(0 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal passer10_result : std_logic_vector(31 downto 0);
  signal passer10_result_valid : std_logic;
  signal passer10_result_ready : std_logic;
  signal passer12_result_valid : std_logic;
  signal passer12_result_ready : std_logic;
  signal buffer9_outs : std_logic_vector(7 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(8 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(31 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(31 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(31 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(31 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(31 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal constant9_outs : std_logic_vector(31 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(8 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(7 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(8 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal mulf4_result : std_logic_vector(31 downto 0);
  signal mulf4_result_valid : std_logic;
  signal mulf4_result_ready : std_logic;
  signal cmpf2_result : std_logic_vector(0 downto 0);
  signal cmpf2_result_valid : std_logic;
  signal cmpf2_result_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(0 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(0 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal andi4_result : std_logic_vector(0 downto 0);
  signal andi4_result_valid : std_logic;
  signal andi4_result_ready : std_logic;
  signal fork42_outs_0 : std_logic_vector(0 downto 0);
  signal fork42_outs_0_valid : std_logic;
  signal fork42_outs_0_ready : std_logic;
  signal fork42_outs_1 : std_logic_vector(0 downto 0);
  signal fork42_outs_1_valid : std_logic;
  signal fork42_outs_1_ready : std_logic;
  signal buffer24_outs : std_logic_vector(8 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork43_outs_0 : std_logic_vector(8 downto 0);
  signal fork43_outs_0_valid : std_logic;
  signal fork43_outs_0_ready : std_logic;
  signal fork43_outs_1 : std_logic_vector(8 downto 0);
  signal fork43_outs_1_valid : std_logic;
  signal fork43_outs_1_ready : std_logic;
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal fork44_outs_0 : std_logic_vector(7 downto 0);
  signal fork44_outs_0_valid : std_logic;
  signal fork44_outs_0_ready : std_logic;
  signal fork44_outs_1 : std_logic_vector(7 downto 0);
  signal fork44_outs_1_valid : std_logic;
  signal fork44_outs_1_ready : std_logic;
  signal trunci0_outs : std_logic_vector(7 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork45_outs_0 : std_logic_vector(0 downto 0);
  signal fork45_outs_0_valid : std_logic;
  signal fork45_outs_0_ready : std_logic;
  signal fork45_outs_1 : std_logic_vector(0 downto 0);
  signal fork45_outs_1_valid : std_logic;
  signal fork45_outs_1_ready : std_logic;
  signal fork45_outs_2 : std_logic_vector(0 downto 0);
  signal fork45_outs_2_valid : std_logic;
  signal fork45_outs_2_ready : std_logic;
  signal andi5_result : std_logic_vector(0 downto 0);
  signal andi5_result_valid : std_logic;
  signal andi5_result_ready : std_logic;
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
  signal fork3_outs_4 : std_logic_vector(0 downto 0);
  signal fork3_outs_4_valid : std_logic;
  signal fork3_outs_4_ready : std_logic;
  signal fork3_outs_5 : std_logic_vector(0 downto 0);
  signal fork3_outs_5_valid : std_logic;
  signal fork3_outs_5_ready : std_logic;
  signal fork3_outs_6 : std_logic_vector(0 downto 0);
  signal fork3_outs_6_valid : std_logic;
  signal fork3_outs_6_ready : std_logic;
  signal buffer25_outs : std_logic_vector(0 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal init6_outs : std_logic_vector(0 downto 0);
  signal init6_outs_valid : std_logic;
  signal init6_outs_ready : std_logic;
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
  signal not2_outs : std_logic_vector(0 downto 0);
  signal not2_outs_valid : std_logic;
  signal not2_outs_ready : std_logic;
  signal passer21_result : std_logic_vector(31 downto 0);
  signal passer21_result_valid : std_logic;
  signal passer21_result_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal passer25_result : std_logic_vector(31 downto 0);
  signal passer25_result_valid : std_logic;
  signal passer25_result_ready : std_logic;
  signal passer27_result : std_logic_vector(31 downto 0);
  signal passer27_result_valid : std_logic;
  signal passer27_result_ready : std_logic;
  signal passer29_result : std_logic_vector(7 downto 0);
  signal passer29_result_valid : std_logic;
  signal passer29_result_ready : std_logic;
  signal passer31_result : std_logic_vector(0 downto 0);
  signal passer31_result_valid : std_logic;
  signal passer31_result_ready : std_logic;
  signal passer33_result_valid : std_logic;
  signal passer33_result_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal passer34_result : std_logic_vector(31 downto 0);
  signal passer34_result_valid : std_logic;
  signal passer34_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal passer35_result : std_logic_vector(31 downto 0);
  signal passer35_result_valid : std_logic;
  signal passer35_result_ready : std_logic;
  signal not3_outs : std_logic_vector(0 downto 0);
  signal not3_outs_valid : std_logic;
  signal not3_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal passer36_result : std_logic_vector(31 downto 0);
  signal passer36_result_valid : std_logic;
  signal passer36_result_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal passer37_result : std_logic_vector(31 downto 0);
  signal passer37_result_valid : std_logic;
  signal passer37_result_ready : std_logic;
  signal passer38_result : std_logic_vector(7 downto 0);
  signal passer38_result_valid : std_logic;
  signal passer38_result_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal passer39_result : std_logic_vector(31 downto 0);
  signal passer39_result_valid : std_logic;
  signal passer39_result_ready : std_logic;
  signal passer40_result : std_logic_vector(31 downto 0);
  signal passer40_result_valid : std_logic;
  signal passer40_result_ready : std_logic;
  signal passer41_result_valid : std_logic;
  signal passer41_result_ready : std_logic;
  signal passer42_result_valid : std_logic;
  signal passer42_result_ready : std_logic;
  signal passer43_result : std_logic_vector(31 downto 0);
  signal passer43_result_valid : std_logic;
  signal passer43_result_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(0 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(0 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork21_outs_2 : std_logic_vector(0 downto 0);
  signal fork21_outs_2_valid : std_logic;
  signal fork21_outs_2_ready : std_logic;
  signal fork21_outs_3 : std_logic_vector(0 downto 0);
  signal fork21_outs_3_valid : std_logic;
  signal fork21_outs_3_ready : std_logic;
  signal fork21_outs_4 : std_logic_vector(0 downto 0);
  signal fork21_outs_4_valid : std_logic;
  signal fork21_outs_4_ready : std_logic;
  signal fork21_outs_5 : std_logic_vector(0 downto 0);
  signal fork21_outs_5_valid : std_logic;
  signal fork21_outs_5_ready : std_logic;
  signal fork21_outs_6 : std_logic_vector(0 downto 0);
  signal fork21_outs_6_valid : std_logic;
  signal fork21_outs_6_ready : std_logic;
  signal cond_br38_trueOut : std_logic_vector(31 downto 0);
  signal cond_br38_trueOut_valid : std_logic;
  signal cond_br38_trueOut_ready : std_logic;
  signal cond_br38_falseOut : std_logic_vector(31 downto 0);
  signal cond_br38_falseOut_valid : std_logic;
  signal cond_br38_falseOut_ready : std_logic;
  signal cond_br39_trueOut : std_logic_vector(31 downto 0);
  signal cond_br39_trueOut_valid : std_logic;
  signal cond_br39_trueOut_ready : std_logic;
  signal cond_br39_falseOut : std_logic_vector(31 downto 0);
  signal cond_br39_falseOut_valid : std_logic;
  signal cond_br39_falseOut_ready : std_logic;
  signal cond_br40_trueOut : std_logic_vector(7 downto 0);
  signal cond_br40_trueOut_valid : std_logic;
  signal cond_br40_trueOut_ready : std_logic;
  signal cond_br40_falseOut : std_logic_vector(7 downto 0);
  signal cond_br40_falseOut_valid : std_logic;
  signal cond_br40_falseOut_ready : std_logic;
  signal cond_br41_trueOut : std_logic_vector(31 downto 0);
  signal cond_br41_trueOut_valid : std_logic;
  signal cond_br41_trueOut_ready : std_logic;
  signal cond_br41_falseOut : std_logic_vector(31 downto 0);
  signal cond_br41_falseOut_valid : std_logic;
  signal cond_br41_falseOut_ready : std_logic;
  signal cond_br42_trueOut : std_logic_vector(31 downto 0);
  signal cond_br42_trueOut_valid : std_logic;
  signal cond_br42_trueOut_ready : std_logic;
  signal cond_br42_falseOut : std_logic_vector(31 downto 0);
  signal cond_br42_falseOut_valid : std_logic;
  signal cond_br42_falseOut_ready : std_logic;
  signal cond_br43_trueOut_valid : std_logic;
  signal cond_br43_trueOut_ready : std_logic;
  signal cond_br43_falseOut_valid : std_logic;
  signal cond_br43_falseOut_ready : std_logic;
  signal cond_br44_trueOut : std_logic_vector(31 downto 0);
  signal cond_br44_trueOut_valid : std_logic;
  signal cond_br44_trueOut_ready : std_logic;
  signal cond_br44_falseOut : std_logic_vector(31 downto 0);
  signal cond_br44_falseOut_valid : std_logic;
  signal cond_br44_falseOut_ready : std_logic;
  signal mux10_outs : std_logic_vector(31 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal control_merge6_outs_valid : std_logic;
  signal control_merge6_outs_ready : std_logic;
  signal control_merge6_index : std_logic_vector(1 downto 0);
  signal control_merge6_index_valid : std_logic;
  signal control_merge6_index_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal control_merge7_outs_valid : std_logic;
  signal control_merge7_outs_ready : std_logic;
  signal control_merge7_index : std_logic_vector(0 downto 0);
  signal control_merge7_index_valid : std_logic;
  signal control_merge7_index_ready : std_logic;

begin

  out0 <= mux11_outs;
  out0_valid <= mux11_outs_valid;
  mux11_outs_ready <= out0_ready;
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

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => a,
      ins_valid => a_valid,
      ins_ready => a_ready,
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

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant5 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => fork1_outs_1,
      lhs_valid => fork1_outs_1_valid,
      lhs_ready => fork1_outs_1_ready,
      rhs => fork1_outs_2,
      rhs_valid => fork1_outs_2_valid,
      rhs_ready => fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => mulf0_result,
      lhs_valid => mulf0_result_valid,
      lhs_ready => mulf0_result_ready,
      rhs => constant5_outs,
      rhs_valid => constant5_outs_valid,
      rhs_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
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
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br38_trueOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br38_trueOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => cond_br38_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork2_outs_2,
      index_valid => fork2_outs_2_valid,
      index_ready => fork2_outs_2_ready,
      ins(0) => b,
      ins(1) => cond_br39_trueOut,
      ins_valid(0) => b_valid,
      ins_valid(1) => cond_br39_trueOut_valid,
      ins_ready(0) => b_ready,
      ins_ready(1) => cond_br39_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => extsi3_outs,
      ins(1) => cond_br40_trueOut,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => cond_br40_trueOut_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => cond_br40_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.handshake_mux_3(arch)
    port map(
      index => fork2_outs_3,
      index_valid => fork2_outs_3_valid,
      index_ready => fork2_outs_3_ready,
      ins(0) => addf0_result,
      ins(1) => cond_br41_trueOut,
      ins_valid(0) => addf0_result_valid,
      ins_valid(1) => cond_br41_trueOut_valid,
      ins_ready(0) => addf0_result_ready,
      ins_ready(1) => cond_br41_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.handshake_mux_4(arch)
    port map(
      index => fork2_outs_4,
      index_valid => fork2_outs_4_valid,
      index_ready => fork2_outs_4_ready,
      ins(0) => tol,
      ins(1) => cond_br42_trueOut,
      ins_valid(0) => tol_valid,
      ins_valid(1) => cond_br42_trueOut_valid,
      ins_ready(0) => tol_ready,
      ins_ready(1) => cond_br42_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br43_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br43_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs(3) => fork2_outs_3,
      outs(4) => fork2_outs_4,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready
    );

  buffer0 : entity work.handshake_buffer_0(arch)
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

  mux5 : entity work.handshake_mux_5(arch)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => buffer0_outs,
      ins(1) => passer36_result,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => passer36_result_valid,
      ins_ready(0) => buffer0_outs_ready,
      ins_ready(1) => passer36_result_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer5 : entity work.handshake_buffer_1(arch)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork47 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork47_outs_0,
      outs(1) => fork47_outs_1,
      outs(2) => fork47_outs_2,
      outs_valid(0) => fork47_outs_0_valid,
      outs_valid(1) => fork47_outs_1_valid,
      outs_valid(2) => fork47_outs_2_valid,
      outs_ready(0) => fork47_outs_0_ready,
      outs_ready(1) => fork47_outs_1_ready,
      outs_ready(2) => fork47_outs_2_ready
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

  mux6 : entity work.handshake_mux_6(arch)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => buffer1_outs,
      ins(1) => passer37_result,
      ins_valid(0) => buffer1_outs_valid,
      ins_valid(1) => passer37_result_valid,
      ins_ready(0) => buffer1_outs_ready,
      ins_ready(1) => passer37_result_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer7 : entity work.handshake_buffer_3(arch)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  fork48 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer7_outs,
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork48_outs_0,
      outs(1) => fork48_outs_1,
      outs(2) => fork48_outs_2,
      outs_valid(0) => fork48_outs_0_valid,
      outs_valid(1) => fork48_outs_1_valid,
      outs_valid(2) => fork48_outs_2_valid,
      outs_ready(0) => fork48_outs_0_ready,
      outs_ready(1) => fork48_outs_1_ready,
      outs_ready(2) => fork48_outs_2_ready
    );

  buffer2 : entity work.handshake_buffer_4(arch)
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

  mux7 : entity work.handshake_mux_2(arch)
    port map(
      index => fork4_outs_2,
      index_valid => fork4_outs_2_valid,
      index_ready => fork4_outs_2_ready,
      ins(0) => buffer2_outs,
      ins(1) => passer38_result,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => passer38_result_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => passer38_result_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  buffer3 : entity work.handshake_buffer_5(arch)
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

  mux8 : entity work.handshake_mux_7(arch)
    port map(
      index => fork4_outs_3,
      index_valid => fork4_outs_3_valid,
      index_ready => fork4_outs_3_ready,
      ins(0) => buffer3_outs,
      ins(1) => passer39_result,
      ins_valid(0) => buffer3_outs_valid,
      ins_valid(1) => passer39_result_valid,
      ins_ready(0) => buffer3_outs_ready,
      ins_ready(1) => passer39_result_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  buffer4 : entity work.handshake_buffer_6(arch)
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

  mux9 : entity work.handshake_mux_8(arch)
    port map(
      index => fork4_outs_4,
      index_valid => fork4_outs_4_valid,
      index_ready => fork4_outs_4_ready,
      ins(0) => buffer4_outs,
      ins(1) => passer40_result,
      ins_valid(0) => buffer4_outs_valid,
      ins_valid(1) => passer40_result_valid,
      ins_ready(0) => buffer4_outs_ready,
      ins_ready(1) => passer40_result_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  buffer11 : entity work.handshake_buffer_7(arch)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.handshake_buffer_8(arch)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  fork49 : entity work.handshake_fork_5(arch)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork49_outs_0,
      outs(1) => fork49_outs_1,
      outs(2) => fork49_outs_2,
      outs(3) => fork49_outs_3,
      outs_valid(0) => fork49_outs_0_valid,
      outs_valid(1) => fork49_outs_1_valid,
      outs_valid(2) => fork49_outs_2_valid,
      outs_valid(3) => fork49_outs_3_valid,
      outs_ready(0) => fork49_outs_0_ready,
      outs_ready(1) => fork49_outs_1_ready,
      outs_ready(2) => fork49_outs_2_ready,
      outs_ready(3) => fork49_outs_3_ready
    );

  mux15 : entity work.handshake_mux_9(arch)
    port map(
      index => fork4_outs_5,
      index_valid => fork4_outs_5_valid,
      index_ready => fork4_outs_5_ready,
      ins_valid(0) => control_merge0_outs_valid,
      ins_valid(1) => passer41_result_valid,
      ins_ready(0) => control_merge0_outs_ready,
      ins_ready(1) => passer41_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  buffer13 : entity work.handshake_buffer_9(arch)
    port map(
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  buffer14 : entity work.handshake_buffer_10(arch)
    port map(
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  fork50 : entity work.handshake_fork_6(arch)
    port map(
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork50_outs_0_valid,
      outs_valid(1) => fork50_outs_1_valid,
      outs_valid(2) => fork50_outs_2_valid,
      outs_valid(3) => fork50_outs_3_valid,
      outs_valid(4) => fork50_outs_4_valid,
      outs_valid(5) => fork50_outs_5_valid,
      outs_ready(0) => fork50_outs_0_ready,
      outs_ready(1) => fork50_outs_1_ready,
      outs_ready(2) => fork50_outs_2_ready,
      outs_ready(3) => fork50_outs_3_ready,
      outs_ready(4) => fork50_outs_4_ready,
      outs_ready(5) => fork50_outs_5_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant6 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant7 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  addf1 : entity work.handshake_addf_1(arch)
    port map(
      lhs => fork47_outs_0,
      lhs_valid => fork47_outs_0_valid,
      lhs_ready => fork47_outs_0_ready,
      rhs => fork48_outs_0,
      rhs_valid => fork48_outs_0_valid,
      rhs_ready => fork48_outs_0_ready,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready
    );

  mulf1 : entity work.handshake_mulf_1(arch)
    port map(
      lhs => addf1_result,
      lhs_valid => addf1_result_valid,
      lhs_ready => addf1_result_ready,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready
    );

  fork51 : entity work.handshake_fork_7(arch)
    port map(
      ins => mulf1_result,
      ins_valid => mulf1_result_valid,
      ins_ready => mulf1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork51_outs_0,
      outs(1) => fork51_outs_1,
      outs(2) => fork51_outs_2,
      outs(3) => fork51_outs_3,
      outs(4) => fork51_outs_4,
      outs(5) => fork51_outs_5,
      outs_valid(0) => fork51_outs_0_valid,
      outs_valid(1) => fork51_outs_1_valid,
      outs_valid(2) => fork51_outs_2_valid,
      outs_valid(3) => fork51_outs_3_valid,
      outs_valid(4) => fork51_outs_4_valid,
      outs_valid(5) => fork51_outs_5_valid,
      outs_ready(0) => fork51_outs_0_ready,
      outs_ready(1) => fork51_outs_1_ready,
      outs_ready(2) => fork51_outs_2_ready,
      outs_ready(3) => fork51_outs_3_ready,
      outs_ready(4) => fork51_outs_4_ready,
      outs_ready(5) => fork51_outs_5_ready
    );

  mulf2 : entity work.handshake_mulf_2(arch)
    port map(
      lhs => fork51_outs_0,
      lhs_valid => fork51_outs_0_valid,
      lhs_ready => fork51_outs_0_ready,
      rhs => fork51_outs_1,
      rhs_valid => fork51_outs_1_valid,
      rhs_ready => fork51_outs_1_ready,
      clk => clk,
      rst => rst,
      result => mulf2_result,
      result_valid => mulf2_result_valid,
      result_ready => mulf2_result_ready
    );

  addf2 : entity work.handshake_addf_2(arch)
    port map(
      lhs => mulf2_result,
      lhs_valid => mulf2_result_valid,
      lhs_ready => mulf2_result_ready,
      rhs => constant6_outs,
      rhs_valid => constant6_outs_valid,
      rhs_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      result => addf2_result,
      result_valid => addf2_result_valid,
      result_ready => addf2_result_ready
    );

  fork52 : entity work.handshake_fork_8(arch)
    port map(
      ins => addf2_result,
      ins_valid => addf2_result_valid,
      ins_ready => addf2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork52_outs_0,
      outs(1) => fork52_outs_1,
      outs(2) => fork52_outs_2,
      outs_valid(0) => fork52_outs_0_valid,
      outs_valid(1) => fork52_outs_1_valid,
      outs_valid(2) => fork52_outs_2_valid,
      outs_ready(0) => fork52_outs_0_ready,
      outs_ready(1) => fork52_outs_1_ready,
      outs_ready(2) => fork52_outs_2_ready
    );

  buffer19 : entity work.handshake_buffer_11(arch)
    port map(
      ins => fork52_outs_0,
      ins_valid => fork52_outs_0_valid,
      ins_ready => fork52_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  absf0 : entity work.handshake_absf_0(arch)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
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
      rhs => fork49_outs_0,
      rhs_valid => fork49_outs_0_valid,
      rhs_ready => fork49_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  fork25 : entity work.handshake_fork_9(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs(2) => fork25_outs_2,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_valid(2) => fork25_outs_2_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready,
      outs_ready(2) => fork25_outs_2_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork25_outs_2,
      ins_valid => fork25_outs_2_valid,
      ins_ready => fork25_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  fork6 : entity work.handshake_fork_10(arch)
    port map(
      ins => not0_outs,
      ins_valid => not0_outs_valid,
      ins_ready => not0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not2_outs,
      lhs_valid => not2_outs_valid,
      lhs_ready => not2_outs_ready,
      rhs => fork17_outs_0,
      rhs_valid => fork17_outs_0_valid,
      rhs_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork7 : entity work.handshake_fork_11(arch)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs(4) => fork7_outs_4,
      outs(5) => fork7_outs_5,
      outs(6) => fork7_outs_6,
      outs(7) => fork7_outs_7,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_valid(4) => fork7_outs_4_valid,
      outs_valid(5) => fork7_outs_5_valid,
      outs_valid(6) => fork7_outs_6_valid,
      outs_valid(7) => fork7_outs_7_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready,
      outs_ready(4) => fork7_outs_4_ready,
      outs_ready(5) => fork7_outs_5_ready,
      outs_ready(6) => fork7_outs_6_ready,
      outs_ready(7) => fork7_outs_7_ready
    );

  andi1 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not3_outs,
      lhs_valid => not3_outs_valid,
      lhs_ready => not3_outs_ready,
      rhs => fork42_outs_0,
      rhs_valid => fork42_outs_0_valid,
      rhs_ready => fork42_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  fork9 : entity work.handshake_fork_10(arch)
    port map(
      ins => andi1_result,
      ins_valid => andi1_result_valid,
      ins_ready => andi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  andi2 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork16_outs_1,
      lhs_valid => fork16_outs_1_valid,
      lhs_ready => fork16_outs_1_ready,
      rhs => fork6_outs_0,
      rhs_valid => fork6_outs_0_valid,
      rhs_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi2_result,
      result_valid => andi2_result_valid,
      result_ready => andi2_result_ready
    );

  fork13 : entity work.handshake_fork_10(arch)
    port map(
      ins => andi2_result,
      ins_valid => andi2_result_valid,
      ins_ready => andi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  buffer15 : entity work.handshake_buffer_12(arch)
    port map(
      ins => fork51_outs_2,
      ins_valid => fork51_outs_2_valid,
      ins_ready => fork51_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  passer0 : entity work.handshake_passer_0(arch)
    port map(
      data => buffer15_outs,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      ctrl => fork25_outs_1,
      ctrl_valid => fork25_outs_1_valid,
      ctrl_ready => fork25_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer0_result,
      result_valid => passer0_result_valid,
      result_ready => passer0_result_ready
    );

  passer2 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork50_outs_0_valid,
      data_ready => fork50_outs_0_ready,
      ctrl => fork25_outs_0,
      ctrl_valid => fork25_outs_0_valid,
      ctrl_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer2_result_valid,
      result_ready => passer2_result_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant8 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  subf0 : entity work.handshake_subf_0(arch)
    port map(
      lhs => fork48_outs_2,
      lhs_valid => fork48_outs_2_valid,
      lhs_ready => fork48_outs_2_ready,
      rhs => fork47_outs_2,
      rhs_valid => fork47_outs_2_valid,
      rhs_ready => fork47_outs_2_ready,
      clk => clk,
      rst => rst,
      result => subf0_result,
      result_valid => subf0_result_valid,
      result_ready => subf0_result_ready
    );

  mulf3 : entity work.handshake_mulf_3(arch)
    port map(
      lhs => subf0_result,
      lhs_valid => subf0_result_valid,
      lhs_ready => subf0_result_ready,
      rhs => constant8_outs,
      rhs_valid => constant8_outs_valid,
      rhs_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf3_result,
      result_valid => mulf3_result_valid,
      result_ready => mulf3_result_ready
    );

  buffer21 : entity work.handshake_buffer_13(arch)
    port map(
      ins => mulf3_result,
      ins_valid => mulf3_result_valid,
      ins_ready => mulf3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  cmpf1 : entity work.handshake_cmpf_1(arch)
    port map(
      lhs => buffer21_outs,
      lhs_valid => buffer21_outs_valid,
      lhs_ready => buffer21_outs_ready,
      rhs => fork49_outs_1,
      rhs_valid => fork49_outs_1_valid,
      rhs_ready => fork49_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpf1_result,
      result_valid => cmpf1_result_valid,
      result_ready => cmpf1_result_ready
    );

  fork16 : entity work.handshake_fork_10(arch)
    port map(
      ins => cmpf1_result,
      ins_valid => cmpf1_result_valid,
      ins_ready => cmpf1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  not1 : entity work.handshake_not_0(arch)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not1_outs,
      outs_valid => not1_outs_valid,
      outs_ready => not1_outs_ready
    );

  andi3 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => not1_outs,
      rhs_valid => not1_outs_valid,
      rhs_ready => not1_outs_ready,
      clk => clk,
      rst => rst,
      result => andi3_result,
      result_valid => andi3_result_valid,
      result_ready => andi3_result_ready
    );

  fork17 : entity work.handshake_fork_10(arch)
    port map(
      ins => andi3_result,
      ins_valid => andi3_result_valid,
      ins_ready => andi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  buffer16 : entity work.handshake_buffer_14(arch)
    port map(
      ins => fork51_outs_3,
      ins_valid => fork51_outs_3_valid,
      ins_ready => fork51_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  passer10 : entity work.handshake_passer_2(arch)
    port map(
      data => buffer16_outs,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      ctrl => fork13_outs_1,
      ctrl_valid => fork13_outs_1_valid,
      ctrl_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer10_result,
      result_valid => passer10_result_valid,
      result_ready => passer10_result_ready
    );

  passer12 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork50_outs_1_valid,
      data_ready => fork50_outs_1_ready,
      ctrl => fork13_outs_0,
      ctrl_valid => fork13_outs_0_valid,
      ctrl_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer12_result_valid,
      result_ready => passer12_result_ready
    );

  buffer9 : entity work.handshake_buffer_4(arch)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  buffer10 : entity work.handshake_buffer_15(arch)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork14 : entity work.handshake_fork_12(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  fork18 : entity work.handshake_fork_13(arch)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready
    );

  constant9 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => fork50_outs_2_valid,
      ctrl_ready => fork50_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant1 : entity work.handshake_constant_6(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
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

  source5 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant2 : entity work.handshake_constant_7(arch)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
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

  buffer23 : entity work.handshake_buffer_16(arch)
    port map(
      ins => fork14_outs_1,
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  mulf4 : entity work.handshake_mulf_4(arch)
    port map(
      lhs => buffer23_outs,
      lhs_valid => buffer23_outs_valid,
      lhs_ready => buffer23_outs_ready,
      rhs => fork52_outs_2,
      rhs_valid => fork52_outs_2_valid,
      rhs_ready => fork52_outs_2_ready,
      clk => clk,
      rst => rst,
      result => mulf4_result,
      result_valid => mulf4_result_valid,
      result_ready => mulf4_result_ready
    );

  cmpf2 : entity work.handshake_cmpf_2(arch)
    port map(
      lhs => mulf4_result,
      lhs_valid => mulf4_result_valid,
      lhs_ready => mulf4_result_ready,
      rhs => fork18_outs_0,
      rhs_valid => fork18_outs_0_valid,
      rhs_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpf2_result,
      result_valid => cmpf2_result_valid,
      result_ready => cmpf2_result_ready
    );

  fork20 : entity work.handshake_fork_10(arch)
    port map(
      ins => cmpf2_result,
      ins_valid => cmpf2_result_valid,
      ins_ready => cmpf2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  andi4 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork17_outs_1,
      lhs_valid => fork17_outs_1_valid,
      lhs_ready => fork17_outs_1_ready,
      rhs => fork20_outs_0,
      rhs_valid => fork20_outs_0_valid,
      rhs_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi4_result,
      result_valid => andi4_result_valid,
      result_ready => andi4_result_ready
    );

  fork42 : entity work.handshake_fork_10(arch)
    port map(
      ins => andi4_result,
      ins_valid => andi4_result_valid,
      ins_ready => andi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork42_outs_0,
      outs(1) => fork42_outs_1,
      outs_valid(0) => fork42_outs_0_valid,
      outs_valid(1) => fork42_outs_1_valid,
      outs_ready(0) => fork42_outs_0_ready,
      outs_ready(1) => fork42_outs_1_ready
    );

  buffer24 : entity work.handshake_buffer_17(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  fork43 : entity work.handshake_fork_14(arch)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork43_outs_0,
      outs(1) => fork43_outs_1,
      outs_valid(0) => fork43_outs_0_valid,
      outs_valid(1) => fork43_outs_1_valid,
      outs_ready(0) => fork43_outs_0_ready,
      outs_ready(1) => fork43_outs_1_ready
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

  fork44 : entity work.handshake_fork_15(arch)
    port map(
      ins => trunci0_outs,
      ins_valid => trunci0_outs_valid,
      ins_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork44_outs_0,
      outs(1) => fork44_outs_1,
      outs_valid(0) => fork44_outs_0_valid,
      outs_valid(1) => fork44_outs_1_valid,
      outs_ready(0) => fork44_outs_0_ready,
      outs_ready(1) => fork44_outs_1_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork43_outs_0,
      ins_valid => fork43_outs_0_valid,
      ins_ready => fork43_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork43_outs_1,
      lhs_valid => fork43_outs_1_valid,
      lhs_ready => fork43_outs_1_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork45 : entity work.handshake_fork_9(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork45_outs_0,
      outs(1) => fork45_outs_1,
      outs(2) => fork45_outs_2,
      outs_valid(0) => fork45_outs_0_valid,
      outs_valid(1) => fork45_outs_1_valid,
      outs_valid(2) => fork45_outs_2_valid,
      outs_ready(0) => fork45_outs_0_ready,
      outs_ready(1) => fork45_outs_1_ready,
      outs_ready(2) => fork45_outs_2_ready
    );

  andi5 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork42_outs_1,
      lhs_valid => fork42_outs_1_valid,
      lhs_ready => fork42_outs_1_ready,
      rhs => fork45_outs_0,
      rhs_valid => fork45_outs_0_valid,
      rhs_ready => fork45_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi5_result,
      result_valid => andi5_result_valid,
      result_ready => andi5_result_ready
    );

  fork3 : entity work.handshake_fork_16(arch)
    port map(
      ins => andi5_result,
      ins_valid => andi5_result_valid,
      ins_ready => andi5_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs(2) => fork3_outs_2,
      outs(3) => fork3_outs_3,
      outs(4) => fork3_outs_4,
      outs(5) => fork3_outs_5,
      outs(6) => fork3_outs_6,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_valid(2) => fork3_outs_2_valid,
      outs_valid(3) => fork3_outs_3_valid,
      outs_valid(4) => fork3_outs_4_valid,
      outs_valid(5) => fork3_outs_5_valid,
      outs_valid(6) => fork3_outs_6_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready,
      outs_ready(2) => fork3_outs_2_ready,
      outs_ready(3) => fork3_outs_3_ready,
      outs_ready(4) => fork3_outs_4_ready,
      outs_ready(5) => fork3_outs_5_ready,
      outs_ready(6) => fork3_outs_6_ready
    );

  buffer25 : entity work.handshake_buffer_18(arch)
    port map(
      ins => fork3_outs_6,
      ins_valid => fork3_outs_6_valid,
      ins_ready => fork3_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  init6 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs => init6_outs,
      outs_valid => init6_outs_valid,
      outs_ready => init6_outs_ready
    );

  fork4 : entity work.handshake_fork_17(arch)
    port map(
      ins => init6_outs,
      ins_valid => init6_outs_valid,
      ins_ready => init6_outs_ready,
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

  not2 : entity work.handshake_not_0(arch)
    port map(
      ins => fork20_outs_1,
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => not2_outs,
      outs_valid => not2_outs_valid,
      outs_ready => not2_outs_ready
    );

  passer21 : entity work.handshake_passer_3(arch)
    port map(
      data => fork49_outs_2,
      data_valid => fork49_outs_2_valid,
      data_ready => fork49_outs_2_ready,
      ctrl => fork7_outs_7,
      ctrl_valid => fork7_outs_7_valid,
      ctrl_ready => fork7_outs_7_ready,
      clk => clk,
      rst => rst,
      result => passer21_result,
      result_valid => passer21_result_valid,
      result_ready => passer21_result_ready
    );

  buffer17 : entity work.handshake_buffer_19(arch)
    port map(
      ins => fork51_outs_4,
      ins_valid => fork51_outs_4_valid,
      ins_ready => fork51_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  passer25 : entity work.handshake_passer_4(arch)
    port map(
      data => buffer17_outs,
      data_valid => buffer17_outs_valid,
      data_ready => buffer17_outs_ready,
      ctrl => fork7_outs_3,
      ctrl_valid => fork7_outs_3_valid,
      ctrl_ready => fork7_outs_3_ready,
      clk => clk,
      rst => rst,
      result => passer25_result,
      result_valid => passer25_result_valid,
      result_ready => passer25_result_ready
    );

  passer27 : entity work.handshake_passer_5(arch)
    port map(
      data => fork18_outs_1,
      data_valid => fork18_outs_1_valid,
      data_ready => fork18_outs_1_ready,
      ctrl => fork7_outs_1,
      ctrl_valid => fork7_outs_1_valid,
      ctrl_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer27_result,
      result_valid => passer27_result_valid,
      result_ready => passer27_result_ready
    );

  passer29 : entity work.handshake_passer_6(arch)
    port map(
      data => fork44_outs_0,
      data_valid => fork44_outs_0_valid,
      data_ready => fork44_outs_0_ready,
      ctrl => fork7_outs_5,
      ctrl_valid => fork7_outs_5_valid,
      ctrl_ready => fork7_outs_5_ready,
      clk => clk,
      rst => rst,
      result => passer29_result,
      result_valid => passer29_result_valid,
      result_ready => passer29_result_ready
    );

  passer31 : entity work.handshake_passer_7(arch)
    port map(
      data => fork45_outs_1,
      data_valid => fork45_outs_1_valid,
      data_ready => fork45_outs_1_ready,
      ctrl => fork7_outs_2,
      ctrl_valid => fork7_outs_2_valid,
      ctrl_ready => fork7_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer31_result,
      result_valid => passer31_result_valid,
      result_ready => passer31_result_ready
    );

  passer33 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork50_outs_3_valid,
      data_ready => fork50_outs_3_ready,
      ctrl => fork7_outs_0,
      ctrl_valid => fork7_outs_0_valid,
      ctrl_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer33_result_valid,
      result_ready => passer33_result_ready
    );

  buffer8 : entity work.handshake_buffer_20(arch)
    port map(
      ins => fork48_outs_1,
      ins_valid => fork48_outs_1_valid,
      ins_ready => fork48_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  passer34 : entity work.handshake_passer_8(arch)
    port map(
      data => buffer8_outs,
      data_valid => buffer8_outs_valid,
      data_ready => buffer8_outs_ready,
      ctrl => fork7_outs_6,
      ctrl_valid => fork7_outs_6_valid,
      ctrl_ready => fork7_outs_6_ready,
      clk => clk,
      rst => rst,
      result => passer34_result,
      result_valid => passer34_result_valid,
      result_ready => passer34_result_ready
    );

  buffer20 : entity work.handshake_buffer_21(arch)
    port map(
      ins => fork52_outs_1,
      ins_valid => fork52_outs_1_valid,
      ins_ready => fork52_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  passer35 : entity work.handshake_passer_9(arch)
    port map(
      data => buffer20_outs,
      data_valid => buffer20_outs_valid,
      data_ready => buffer20_outs_ready,
      ctrl => fork7_outs_4,
      ctrl_valid => fork7_outs_4_valid,
      ctrl_ready => fork7_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer35_result,
      result_valid => passer35_result_valid,
      result_ready => passer35_result_ready
    );

  not3 : entity work.handshake_not_0(arch)
    port map(
      ins => fork45_outs_2,
      ins_valid => fork45_outs_2_valid,
      ins_ready => fork45_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => not3_outs,
      outs_valid => not3_outs_valid,
      outs_ready => not3_outs_ready
    );

  buffer6 : entity work.handshake_buffer_22(arch)
    port map(
      ins => fork47_outs_1,
      ins_valid => fork47_outs_1_valid,
      ins_ready => fork47_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  passer36 : entity work.handshake_passer_10(arch)
    port map(
      data => buffer6_outs,
      data_valid => buffer6_outs_valid,
      data_ready => buffer6_outs_ready,
      ctrl => fork3_outs_4,
      ctrl_valid => fork3_outs_4_valid,
      ctrl_ready => fork3_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer36_result,
      result_valid => passer36_result_valid,
      result_ready => passer36_result_ready
    );

  buffer18 : entity work.handshake_buffer_23(arch)
    port map(
      ins => fork51_outs_5,
      ins_valid => fork51_outs_5_valid,
      ins_ready => fork51_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  passer37 : entity work.handshake_passer_11(arch)
    port map(
      data => buffer18_outs,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      ctrl => fork3_outs_5,
      ctrl_valid => fork3_outs_5_valid,
      ctrl_ready => fork3_outs_5_ready,
      clk => clk,
      rst => rst,
      result => passer37_result,
      result_valid => passer37_result_valid,
      result_ready => passer37_result_ready
    );

  passer38 : entity work.handshake_passer_6(arch)
    port map(
      data => fork44_outs_1,
      data_valid => fork44_outs_1_valid,
      data_ready => fork44_outs_1_ready,
      ctrl => fork3_outs_0,
      ctrl_valid => fork3_outs_0_valid,
      ctrl_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer38_result,
      result_valid => passer38_result_valid,
      result_ready => passer38_result_ready
    );

  buffer22 : entity work.handshake_buffer_24(arch)
    port map(
      ins => fork14_outs_0,
      ins_valid => fork14_outs_0_valid,
      ins_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  passer39 : entity work.handshake_passer_12(arch)
    port map(
      data => buffer22_outs,
      data_valid => buffer22_outs_valid,
      data_ready => buffer22_outs_ready,
      ctrl => fork3_outs_3,
      ctrl_valid => fork3_outs_3_valid,
      ctrl_ready => fork3_outs_3_ready,
      clk => clk,
      rst => rst,
      result => passer39_result,
      result_valid => passer39_result_valid,
      result_ready => passer39_result_ready
    );

  passer40 : entity work.handshake_passer_13(arch)
    port map(
      data => fork49_outs_3,
      data_valid => fork49_outs_3_valid,
      data_ready => fork49_outs_3_ready,
      ctrl => fork3_outs_1,
      ctrl_valid => fork3_outs_1_valid,
      ctrl_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer40_result,
      result_valid => passer40_result_valid,
      result_ready => passer40_result_ready
    );

  passer41 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork50_outs_5_valid,
      data_ready => fork50_outs_5_ready,
      ctrl => fork3_outs_2,
      ctrl_valid => fork3_outs_2_valid,
      ctrl_ready => fork3_outs_2_ready,
      clk => clk,
      rst => rst,
      result_valid => passer41_result_valid,
      result_ready => passer41_result_ready
    );

  passer42 : entity work.handshake_passer_1(arch)
    port map(
      data_valid => fork50_outs_4_valid,
      data_ready => fork50_outs_4_ready,
      ctrl => fork9_outs_0,
      ctrl_valid => fork9_outs_0_valid,
      ctrl_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer42_result_valid,
      result_ready => passer42_result_ready
    );

  passer43 : entity work.handshake_passer_14(arch)
    port map(
      data => fork18_outs_2,
      data_valid => fork18_outs_2_valid,
      data_ready => fork18_outs_2_ready,
      ctrl => fork9_outs_1,
      ctrl_valid => fork9_outs_1_valid,
      ctrl_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer43_result,
      result_valid => passer43_result_valid,
      result_ready => passer43_result_ready
    );

  fork21 : entity work.handshake_fork_16(arch)
    port map(
      ins => passer31_result,
      ins_valid => passer31_result_valid,
      ins_ready => passer31_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs(2) => fork21_outs_2,
      outs(3) => fork21_outs_3,
      outs(4) => fork21_outs_4,
      outs(5) => fork21_outs_5,
      outs(6) => fork21_outs_6,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_valid(2) => fork21_outs_2_valid,
      outs_valid(3) => fork21_outs_3_valid,
      outs_valid(4) => fork21_outs_4_valid,
      outs_valid(5) => fork21_outs_5_valid,
      outs_valid(6) => fork21_outs_6_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready,
      outs_ready(2) => fork21_outs_2_ready,
      outs_ready(3) => fork21_outs_3_ready,
      outs_ready(4) => fork21_outs_4_ready,
      outs_ready(5) => fork21_outs_5_ready,
      outs_ready(6) => fork21_outs_6_ready
    );

  cond_br38 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork21_outs_6,
      condition_valid => fork21_outs_6_valid,
      condition_ready => fork21_outs_6_ready,
      data => passer25_result,
      data_valid => passer25_result_valid,
      data_ready => passer25_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br38_trueOut,
      trueOut_valid => cond_br38_trueOut_valid,
      trueOut_ready => cond_br38_trueOut_ready,
      falseOut => cond_br38_falseOut,
      falseOut_valid => cond_br38_falseOut_valid,
      falseOut_ready => cond_br38_falseOut_ready
    );

  sink26 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br38_falseOut,
      ins_valid => cond_br38_falseOut_valid,
      ins_ready => cond_br38_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br39 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork21_outs_5,
      condition_valid => fork21_outs_5_valid,
      condition_ready => fork21_outs_5_ready,
      data => passer34_result,
      data_valid => passer34_result_valid,
      data_ready => passer34_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br39_trueOut,
      trueOut_valid => cond_br39_trueOut_valid,
      trueOut_ready => cond_br39_trueOut_ready,
      falseOut => cond_br39_falseOut,
      falseOut_valid => cond_br39_falseOut_valid,
      falseOut_ready => cond_br39_falseOut_ready
    );

  sink27 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br39_falseOut,
      ins_valid => cond_br39_falseOut_valid,
      ins_ready => cond_br39_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br40 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork21_outs_0,
      condition_valid => fork21_outs_0_valid,
      condition_ready => fork21_outs_0_ready,
      data => passer29_result,
      data_valid => passer29_result_valid,
      data_ready => passer29_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br40_trueOut,
      trueOut_valid => cond_br40_trueOut_valid,
      trueOut_ready => cond_br40_trueOut_ready,
      falseOut => cond_br40_falseOut,
      falseOut_valid => cond_br40_falseOut_valid,
      falseOut_ready => cond_br40_falseOut_ready
    );

  sink28 : entity work.handshake_sink_2(arch)
    port map(
      ins => cond_br40_falseOut,
      ins_valid => cond_br40_falseOut_valid,
      ins_ready => cond_br40_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br41 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork21_outs_4,
      condition_valid => fork21_outs_4_valid,
      condition_ready => fork21_outs_4_ready,
      data => passer35_result,
      data_valid => passer35_result_valid,
      data_ready => passer35_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br41_trueOut,
      trueOut_valid => cond_br41_trueOut_valid,
      trueOut_ready => cond_br41_trueOut_ready,
      falseOut => cond_br41_falseOut,
      falseOut_valid => cond_br41_falseOut_valid,
      falseOut_ready => cond_br41_falseOut_ready
    );

  sink29 : entity work.handshake_sink_3(arch)
    port map(
      ins => cond_br41_falseOut,
      ins_valid => cond_br41_falseOut_valid,
      ins_ready => cond_br41_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br42 : entity work.handshake_cond_br_4(arch)
    port map(
      condition => fork21_outs_3,
      condition_valid => fork21_outs_3_valid,
      condition_ready => fork21_outs_3_ready,
      data => passer21_result,
      data_valid => passer21_result_valid,
      data_ready => passer21_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br42_trueOut,
      trueOut_valid => cond_br42_trueOut_valid,
      trueOut_ready => cond_br42_trueOut_ready,
      falseOut => cond_br42_falseOut,
      falseOut_valid => cond_br42_falseOut_valid,
      falseOut_ready => cond_br42_falseOut_ready
    );

  sink30 : entity work.handshake_sink_4(arch)
    port map(
      ins => cond_br42_falseOut,
      ins_valid => cond_br42_falseOut_valid,
      ins_ready => cond_br42_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br43 : entity work.handshake_cond_br_5(arch)
    port map(
      condition => fork21_outs_2,
      condition_valid => fork21_outs_2_valid,
      condition_ready => fork21_outs_2_ready,
      data_valid => passer33_result_valid,
      data_ready => passer33_result_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br43_trueOut_valid,
      trueOut_ready => cond_br43_trueOut_ready,
      falseOut_valid => cond_br43_falseOut_valid,
      falseOut_ready => cond_br43_falseOut_ready
    );

  cond_br44 : entity work.handshake_cond_br_6(arch)
    port map(
      condition => fork21_outs_1,
      condition_valid => fork21_outs_1_valid,
      condition_ready => fork21_outs_1_ready,
      data => passer27_result,
      data_valid => passer27_result_valid,
      data_ready => passer27_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br44_trueOut,
      trueOut_valid => cond_br44_trueOut_valid,
      trueOut_ready => cond_br44_trueOut_ready,
      falseOut => cond_br44_falseOut,
      falseOut_valid => cond_br44_falseOut_valid,
      falseOut_ready => cond_br44_falseOut_ready
    );

  sink31 : entity work.handshake_sink_5(arch)
    port map(
      ins => cond_br44_trueOut,
      ins_valid => cond_br44_trueOut_valid,
      ins_ready => cond_br44_trueOut_ready,
      clk => clk,
      rst => rst
    );

  mux10 : entity work.handshake_mux_10(arch)
    port map(
      index => control_merge6_index,
      index_valid => control_merge6_index_valid,
      index_ready => control_merge6_index_ready,
      ins(0) => passer0_result,
      ins(1) => passer43_result,
      ins(2) => cond_br44_falseOut,
      ins_valid(0) => passer0_result_valid,
      ins_valid(1) => passer43_result_valid,
      ins_valid(2) => cond_br44_falseOut_valid,
      ins_ready(0) => passer0_result_ready,
      ins_ready(1) => passer43_result_ready,
      ins_ready(2) => cond_br44_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  control_merge6 : entity work.handshake_control_merge_1(arch)
    port map(
      ins_valid(0) => passer2_result_valid,
      ins_valid(1) => passer42_result_valid,
      ins_valid(2) => cond_br43_falseOut_valid,
      ins_ready(0) => passer2_result_ready,
      ins_ready(1) => passer42_result_ready,
      ins_ready(2) => cond_br43_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge6_outs_valid,
      outs_ready => control_merge6_outs_ready,
      index => control_merge6_index,
      index_valid => control_merge6_index_valid,
      index_ready => control_merge6_index_ready
    );

  mux11 : entity work.handshake_mux_11(arch)
    port map(
      index => control_merge7_index,
      index_valid => control_merge7_index_valid,
      index_ready => control_merge7_index_ready,
      ins(0) => passer10_result,
      ins(1) => mux10_outs,
      ins_valid(0) => passer10_result_valid,
      ins_valid(1) => mux10_outs_valid,
      ins_ready(0) => passer10_result_ready,
      ins_ready(1) => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  control_merge7 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => passer12_result_valid,
      ins_valid(1) => control_merge6_outs_valid,
      ins_ready(0) => passer12_result_ready,
      ins_ready(1) => control_merge6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge7_outs_valid,
      outs_ready => control_merge7_outs_ready,
      index => control_merge7_index,
      index_valid => control_merge7_index_valid,
      index_ready => control_merge7_index_ready
    );

  sink32 : entity work.handshake_sink_6(arch)
    port map(
      ins_valid => control_merge7_outs_valid,
      ins_ready => control_merge7_outs_ready,
      clk => clk,
      rst => rst
    );

end architecture;
