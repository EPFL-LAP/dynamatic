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
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(31 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(31 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal mux2_outs : std_logic_vector(7 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(31 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(31 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_2 : std_logic_vector(0 downto 0);
  signal fork5_outs_2_valid : std_logic;
  signal fork5_outs_2_ready : std_logic;
  signal fork5_outs_3 : std_logic_vector(0 downto 0);
  signal fork5_outs_3_valid : std_logic;
  signal fork5_outs_3_ready : std_logic;
  signal fork5_outs_4 : std_logic_vector(0 downto 0);
  signal fork5_outs_4_valid : std_logic;
  signal fork5_outs_4_ready : std_logic;
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
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(31 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal mulf2_result : std_logic_vector(31 downto 0);
  signal mulf2_result_valid : std_logic;
  signal mulf2_result_ready : std_logic;
  signal addf2_result : std_logic_vector(31 downto 0);
  signal addf2_result_valid : std_logic;
  signal addf2_result_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal absf0_outs : std_logic_vector(31 downto 0);
  signal absf0_outs_valid : std_logic;
  signal absf0_outs_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(0 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(0 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(0 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(0 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal fork8_outs_4 : std_logic_vector(0 downto 0);
  signal fork8_outs_4_valid : std_logic;
  signal fork8_outs_4_ready : std_logic;
  signal fork8_outs_5 : std_logic_vector(0 downto 0);
  signal fork8_outs_5_valid : std_logic;
  signal fork8_outs_5_ready : std_logic;
  signal fork8_outs_6 : std_logic_vector(0 downto 0);
  signal fork8_outs_6_valid : std_logic;
  signal fork8_outs_6_ready : std_logic;
  signal fork8_outs_7 : std_logic_vector(0 downto 0);
  signal fork8_outs_7_valid : std_logic;
  signal fork8_outs_7_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(0 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(31 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(31 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(31 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(31 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(31 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(31 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer4_outs : std_logic_vector(7 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(7 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(7 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(7 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(31 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(31 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(31 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(31 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
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
  signal cmpf1_result : std_logic_vector(0 downto 0);
  signal cmpf1_result_valid : std_logic;
  signal cmpf1_result_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal fork12_outs_2 : std_logic_vector(0 downto 0);
  signal fork12_outs_2_valid : std_logic;
  signal fork12_outs_2_ready : std_logic;
  signal fork12_outs_3 : std_logic_vector(0 downto 0);
  signal fork12_outs_3_valid : std_logic;
  signal fork12_outs_3_ready : std_logic;
  signal fork12_outs_4 : std_logic_vector(0 downto 0);
  signal fork12_outs_4_valid : std_logic;
  signal fork12_outs_4_ready : std_logic;
  signal fork12_outs_5 : std_logic_vector(0 downto 0);
  signal fork12_outs_5_valid : std_logic;
  signal fork12_outs_5_ready : std_logic;
  signal fork12_outs_6 : std_logic_vector(0 downto 0);
  signal fork12_outs_6_valid : std_logic;
  signal fork12_outs_6_ready : std_logic;
  signal fork12_outs_7 : std_logic_vector(0 downto 0);
  signal fork12_outs_7_valid : std_logic;
  signal fork12_outs_7_ready : std_logic;
  signal cond_br11_trueOut : std_logic_vector(31 downto 0);
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut : std_logic_vector(31 downto 0);
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal buffer20_outs : std_logic_vector(0 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal cond_br13_trueOut : std_logic_vector(31 downto 0);
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut : std_logic_vector(31 downto 0);
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal cond_br14_trueOut : std_logic_vector(31 downto 0);
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut : std_logic_vector(31 downto 0);
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal cond_br15_trueOut : std_logic_vector(31 downto 0);
  signal cond_br15_trueOut_valid : std_logic;
  signal cond_br15_trueOut_ready : std_logic;
  signal cond_br15_falseOut : std_logic_vector(31 downto 0);
  signal cond_br15_falseOut_valid : std_logic;
  signal cond_br15_falseOut_ready : std_logic;
  signal cond_br16_trueOut : std_logic_vector(7 downto 0);
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut : std_logic_vector(7 downto 0);
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal cond_br17_trueOut : std_logic_vector(31 downto 0);
  signal cond_br17_trueOut_valid : std_logic;
  signal cond_br17_trueOut_ready : std_logic;
  signal cond_br17_falseOut : std_logic_vector(31 downto 0);
  signal cond_br17_falseOut_valid : std_logic;
  signal cond_br17_falseOut_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal cond_br18_trueOut : std_logic_vector(31 downto 0);
  signal cond_br18_trueOut_valid : std_logic;
  signal cond_br18_trueOut_ready : std_logic;
  signal cond_br18_falseOut : std_logic_vector(31 downto 0);
  signal cond_br18_falseOut_valid : std_logic;
  signal cond_br18_falseOut_ready : std_logic;
  signal extsi4_outs : std_logic_vector(8 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(31 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(31 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal constant9_outs : std_logic_vector(31 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(31 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(31 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
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
  signal mulf4_result : std_logic_vector(31 downto 0);
  signal mulf4_result_valid : std_logic;
  signal mulf4_result_ready : std_logic;
  signal cmpf2_result : std_logic_vector(0 downto 0);
  signal cmpf2_result_valid : std_logic;
  signal cmpf2_result_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(0 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(0 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(0 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal buffer24_outs : std_logic_vector(31 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal select0_result : std_logic_vector(31 downto 0);
  signal select0_result_valid : std_logic;
  signal select0_result_ready : std_logic;
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal select1_result : std_logic_vector(31 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;
  signal buffer25_outs : std_logic_vector(31 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(31 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal select2_result : std_logic_vector(31 downto 0);
  signal select2_result_valid : std_logic;
  signal select2_result_ready : std_logic;
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal buffer27_outs : std_logic_vector(8 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(8 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(8 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal trunci0_outs : std_logic_vector(7 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(0 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(0 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal fork20_outs_2 : std_logic_vector(0 downto 0);
  signal fork20_outs_2_valid : std_logic;
  signal fork20_outs_2_ready : std_logic;
  signal fork20_outs_3 : std_logic_vector(0 downto 0);
  signal fork20_outs_3_valid : std_logic;
  signal fork20_outs_3_ready : std_logic;
  signal fork20_outs_4 : std_logic_vector(0 downto 0);
  signal fork20_outs_4_valid : std_logic;
  signal fork20_outs_4_ready : std_logic;
  signal fork20_outs_5 : std_logic_vector(0 downto 0);
  signal fork20_outs_5_valid : std_logic;
  signal fork20_outs_5_ready : std_logic;
  signal fork20_outs_6 : std_logic_vector(0 downto 0);
  signal fork20_outs_6_valid : std_logic;
  signal fork20_outs_6_ready : std_logic;
  signal cond_br19_trueOut : std_logic_vector(31 downto 0);
  signal cond_br19_trueOut_valid : std_logic;
  signal cond_br19_trueOut_ready : std_logic;
  signal cond_br19_falseOut : std_logic_vector(31 downto 0);
  signal cond_br19_falseOut_valid : std_logic;
  signal cond_br19_falseOut_ready : std_logic;
  signal cond_br20_trueOut : std_logic_vector(31 downto 0);
  signal cond_br20_trueOut_valid : std_logic;
  signal cond_br20_trueOut_ready : std_logic;
  signal cond_br20_falseOut : std_logic_vector(31 downto 0);
  signal cond_br20_falseOut_valid : std_logic;
  signal cond_br20_falseOut_ready : std_logic;
  signal cond_br21_trueOut : std_logic_vector(7 downto 0);
  signal cond_br21_trueOut_valid : std_logic;
  signal cond_br21_trueOut_ready : std_logic;
  signal cond_br21_falseOut : std_logic_vector(7 downto 0);
  signal cond_br21_falseOut_valid : std_logic;
  signal cond_br21_falseOut_ready : std_logic;
  signal cond_br22_trueOut : std_logic_vector(31 downto 0);
  signal cond_br22_trueOut_valid : std_logic;
  signal cond_br22_trueOut_ready : std_logic;
  signal cond_br22_falseOut : std_logic_vector(31 downto 0);
  signal cond_br22_falseOut_valid : std_logic;
  signal cond_br22_falseOut_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal cond_br23_trueOut : std_logic_vector(31 downto 0);
  signal cond_br23_trueOut_valid : std_logic;
  signal cond_br23_trueOut_ready : std_logic;
  signal cond_br23_falseOut : std_logic_vector(31 downto 0);
  signal cond_br23_falseOut_valid : std_logic;
  signal cond_br23_falseOut_ready : std_logic;
  signal cond_br24_trueOut_valid : std_logic;
  signal cond_br24_trueOut_ready : std_logic;
  signal cond_br24_falseOut_valid : std_logic;
  signal cond_br24_falseOut_ready : std_logic;
  signal cond_br25_trueOut : std_logic_vector(31 downto 0);
  signal cond_br25_trueOut_valid : std_logic;
  signal cond_br25_trueOut_ready : std_logic;
  signal cond_br25_falseOut : std_logic_vector(31 downto 0);
  signal cond_br25_falseOut_valid : std_logic;
  signal cond_br25_falseOut_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal control_merge3_outs_valid : std_logic;
  signal control_merge3_outs_ready : std_logic;
  signal control_merge3_index : std_logic_vector(0 downto 0);
  signal control_merge3_index_valid : std_logic;
  signal control_merge3_index_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal control_merge4_outs_valid : std_logic;
  signal control_merge4_outs_ready : std_logic;
  signal control_merge4_index : std_logic_vector(0 downto 0);
  signal control_merge4_index_valid : std_logic;
  signal control_merge4_index_ready : std_logic;

begin

  out0 <= mux6_outs;
  out0_valid <= mux6_outs_valid;
  mux6_outs_ready <= out0_ready;
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
      index => fork5_outs_1,
      index_valid => fork5_outs_1_valid,
      index_ready => fork5_outs_1_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br19_trueOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br19_trueOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => cond_br19_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
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

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork5_outs_2,
      index_valid => fork5_outs_2_valid,
      index_ready => fork5_outs_2_ready,
      ins(0) => b,
      ins(1) => cond_br20_trueOut,
      ins_valid(0) => b_valid,
      ins_valid(1) => cond_br20_trueOut_valid,
      ins_ready(0) => b_ready,
      ins_ready(1) => cond_br20_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer2 : entity work.handshake_buffer_1(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  fork3 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => fork5_outs_0,
      index_valid => fork5_outs_0_valid,
      index_ready => fork5_outs_0_ready,
      ins(0) => extsi3_outs,
      ins(1) => cond_br21_trueOut,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => cond_br21_trueOut_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => cond_br21_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.handshake_mux_3(arch)
    port map(
      index => fork5_outs_3,
      index_valid => fork5_outs_3_valid,
      index_ready => fork5_outs_3_ready,
      ins(0) => addf0_result,
      ins(1) => cond_br22_trueOut,
      ins_valid(0) => addf0_result_valid,
      ins_valid(1) => cond_br22_trueOut_valid,
      ins_ready(0) => addf0_result_ready,
      ins_ready(1) => cond_br22_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.handshake_mux_4(arch)
    port map(
      index => fork5_outs_4,
      index_valid => fork5_outs_4_valid,
      index_ready => fork5_outs_4_ready,
      ins(0) => tol,
      ins(1) => cond_br23_trueOut,
      ins_valid(0) => tol_valid,
      ins_valid(1) => cond_br23_trueOut_valid,
      ins_ready(0) => tol_ready,
      ins_ready(1) => cond_br23_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer8 : entity work.handshake_buffer_2(arch)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.handshake_buffer_3(arch)
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

  fork4 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br24_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br24_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork5 : entity work.handshake_fork_5(arch)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs(2) => fork5_outs_2,
      outs(3) => fork5_outs_3,
      outs(4) => fork5_outs_4,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_valid(3) => fork5_outs_3_valid,
      outs_valid(4) => fork5_outs_4_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready,
      outs_ready(3) => fork5_outs_3_ready,
      outs_ready(4) => fork5_outs_4_ready
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
      lhs => fork2_outs_1,
      lhs_valid => fork2_outs_1_valid,
      lhs_ready => fork2_outs_1_ready,
      rhs => fork3_outs_1,
      rhs_valid => fork3_outs_1_valid,
      rhs_ready => fork3_outs_1_ready,
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

  fork6 : entity work.handshake_fork_6(arch)
    port map(
      ins => mulf1_result,
      ins_valid => mulf1_result_valid,
      ins_ready => mulf1_result_ready,
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

  mulf2 : entity work.handshake_mulf_2(arch)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => fork6_outs_2,
      rhs_valid => fork6_outs_2_valid,
      rhs_ready => fork6_outs_2_ready,
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

  fork7 : entity work.handshake_fork_7(arch)
    port map(
      ins => addf2_result,
      ins_valid => addf2_result_valid,
      ins_ready => addf2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  absf0 : entity work.handshake_absf_0(arch)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
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
      rhs => fork4_outs_1,
      rhs_valid => fork4_outs_1_valid,
      rhs_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  fork8 : entity work.handshake_fork_8(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs(4) => fork8_outs_4,
      outs(5) => fork8_outs_5,
      outs(6) => fork8_outs_6,
      outs(7) => fork8_outs_7,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_valid(4) => fork8_outs_4_valid,
      outs_valid(5) => fork8_outs_5_valid,
      outs_valid(6) => fork8_outs_6_valid,
      outs_valid(7) => fork8_outs_7_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready,
      outs_ready(4) => fork8_outs_4_ready,
      outs_ready(5) => fork8_outs_5_ready,
      outs_ready(6) => fork8_outs_6_ready,
      outs_ready(7) => fork8_outs_7_ready
    );

  buffer11 : entity work.handshake_buffer_4(arch)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer14 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork8_outs_7,
      ins_valid => fork8_outs_7_valid,
      ins_ready => fork8_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  cond_br3 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => buffer14_outs,
      condition_valid => buffer14_outs_valid,
      condition_ready => buffer14_outs_ready,
      data => buffer11_outs,
      data_valid => buffer11_outs_valid,
      data_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer10 : entity work.handshake_buffer_6(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer13 : entity work.handshake_buffer_7(arch)
    port map(
      ins => fork8_outs_6,
      ins_valid => fork8_outs_6_valid,
      ins_ready => fork8_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  cond_br4 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer13_outs,
      condition_valid => buffer13_outs_valid,
      condition_ready => buffer13_outs_ready,
      data_valid => buffer10_outs_valid,
      data_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  cond_br5 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork8_outs_5,
      condition_valid => fork8_outs_5_valid,
      condition_ready => fork8_outs_5_ready,
      data => fork4_outs_0,
      data_valid => fork4_outs_0_valid,
      data_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  sink0 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br5_trueOut,
      ins_valid => cond_br5_trueOut_valid,
      ins_ready => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer1 : entity work.handshake_buffer_8(arch)
    port map(
      ins => fork2_outs_0,
      ins_valid => fork2_outs_0_valid,
      ins_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  cond_br6 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork8_outs_4,
      condition_valid => fork8_outs_4_valid,
      condition_ready => fork8_outs_4_ready,
      data => buffer1_outs,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink1 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br6_trueOut,
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer3 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  cond_br7 : entity work.handshake_cond_br_4(arch)
    port map(
      condition => fork8_outs_3,
      condition_valid => fork8_outs_3_valid,
      condition_ready => fork8_outs_3_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink2 : entity work.handshake_sink_2(arch)
    port map(
      ins => cond_br7_trueOut,
      ins_valid => cond_br7_trueOut_valid,
      ins_ready => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer4 : entity work.handshake_buffer_10(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.handshake_buffer_11(arch)
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

  buffer12 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  cond_br8 : entity work.handshake_cond_br_5(arch)
    port map(
      condition => buffer12_outs,
      condition_valid => buffer12_outs_valid,
      condition_ready => buffer12_outs_ready,
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink3 : entity work.handshake_sink_3(arch)
    port map(
      ins => cond_br8_trueOut,
      ins_valid => cond_br8_trueOut_valid,
      ins_ready => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer6 : entity work.handshake_buffer_12(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer7 : entity work.handshake_buffer_13(arch)
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

  cond_br9 : entity work.handshake_cond_br_6(arch)
    port map(
      condition => fork8_outs_2,
      condition_valid => fork8_outs_2_valid,
      condition_ready => fork8_outs_2_ready,
      data => buffer7_outs,
      data_valid => buffer7_outs_valid,
      data_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  sink4 : entity work.handshake_sink_4(arch)
    port map(
      ins => cond_br9_trueOut,
      ins_valid => cond_br9_trueOut_valid,
      ins_ready => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br10 : entity work.handshake_cond_br_7(arch)
    port map(
      condition => fork8_outs_1,
      condition_valid => fork8_outs_1_valid,
      condition_ready => fork8_outs_1_ready,
      data => fork7_outs_0,
      data_valid => fork7_outs_0_valid,
      data_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink5 : entity work.handshake_sink_5(arch)
    port map(
      ins => cond_br10_trueOut,
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer15 : entity work.handshake_buffer_14(arch)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  fork9 : entity work.handshake_fork_9(arch)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  fork10 : entity work.handshake_fork_10(arch)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  fork11 : entity work.handshake_fork_11(arch)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
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
      lhs => fork11_outs_1,
      lhs_valid => fork11_outs_1_valid,
      lhs_ready => fork11_outs_1_ready,
      rhs => fork10_outs_1,
      rhs_valid => fork10_outs_1_valid,
      rhs_ready => fork10_outs_1_ready,
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

  cmpf1 : entity work.handshake_cmpf_1(arch)
    port map(
      lhs => mulf3_result,
      lhs_valid => mulf3_result_valid,
      lhs_ready => mulf3_result_ready,
      rhs => fork9_outs_1,
      rhs_valid => fork9_outs_1_valid,
      rhs_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpf1_result,
      result_valid => cmpf1_result_valid,
      result_ready => cmpf1_result_ready
    );

  fork12 : entity work.handshake_fork_8(arch)
    port map(
      ins => cmpf1_result,
      ins_valid => cmpf1_result_valid,
      ins_ready => cmpf1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs(3) => fork12_outs_3,
      outs(4) => fork12_outs_4,
      outs(5) => fork12_outs_5,
      outs(6) => fork12_outs_6,
      outs(7) => fork12_outs_7,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_valid(3) => fork12_outs_3_valid,
      outs_valid(4) => fork12_outs_4_valid,
      outs_valid(5) => fork12_outs_5_valid,
      outs_valid(6) => fork12_outs_6_valid,
      outs_valid(7) => fork12_outs_7_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready,
      outs_ready(3) => fork12_outs_3_ready,
      outs_ready(4) => fork12_outs_4_ready,
      outs_ready(5) => fork12_outs_5_ready,
      outs_ready(6) => fork12_outs_6_ready,
      outs_ready(7) => fork12_outs_7_ready
    );

  cond_br11 : entity work.handshake_cond_br_8(arch)
    port map(
      condition => fork12_outs_7,
      condition_valid => fork12_outs_7_valid,
      condition_ready => fork12_outs_7_ready,
      data => cond_br3_falseOut,
      data_valid => cond_br3_falseOut_valid,
      data_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br11_trueOut,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut => cond_br11_falseOut,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  buffer20 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork12_outs_6,
      ins_valid => fork12_outs_6_valid,
      ins_ready => fork12_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cond_br12 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer20_outs,
      condition_valid => buffer20_outs_valid,
      condition_ready => buffer20_outs_ready,
      data_valid => cond_br4_falseOut_valid,
      data_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  cond_br13 : entity work.handshake_cond_br_9(arch)
    port map(
      condition => fork12_outs_5,
      condition_valid => fork12_outs_5_valid,
      condition_ready => fork12_outs_5_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  sink7 : entity work.handshake_sink_6(arch)
    port map(
      ins => cond_br13_trueOut,
      ins_valid => cond_br13_trueOut_valid,
      ins_ready => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer18 : entity work.handshake_buffer_15(arch)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  cond_br14 : entity work.handshake_cond_br_10(arch)
    port map(
      condition => fork12_outs_4,
      condition_valid => fork12_outs_4_valid,
      condition_ready => fork12_outs_4_ready,
      data => buffer18_outs,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br14_trueOut,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut => cond_br14_falseOut,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  sink8 : entity work.handshake_sink_7(arch)
    port map(
      ins => cond_br14_trueOut,
      ins_valid => cond_br14_trueOut_valid,
      ins_ready => cond_br14_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer19 : entity work.handshake_buffer_16(arch)
    port map(
      ins => fork11_outs_0,
      ins_valid => fork11_outs_0_valid,
      ins_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  cond_br15 : entity work.handshake_cond_br_11(arch)
    port map(
      condition => fork12_outs_3,
      condition_valid => fork12_outs_3_valid,
      condition_ready => fork12_outs_3_ready,
      data => buffer19_outs,
      data_valid => buffer19_outs_valid,
      data_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  sink9 : entity work.handshake_sink_8(arch)
    port map(
      ins => cond_br15_trueOut,
      ins_valid => cond_br15_trueOut_valid,
      ins_ready => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br16 : entity work.handshake_cond_br_5(arch)
    port map(
      condition => fork12_outs_0,
      condition_valid => fork12_outs_0_valid,
      condition_ready => fork12_outs_0_ready,
      data => cond_br8_falseOut,
      data_valid => cond_br8_falseOut_valid,
      data_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br16_trueOut,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut => cond_br16_falseOut,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  sink10 : entity work.handshake_sink_3(arch)
    port map(
      ins => cond_br16_trueOut,
      ins_valid => cond_br16_trueOut_valid,
      ins_ready => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer16 : entity work.handshake_buffer_17(arch)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  cond_br17 : entity work.handshake_cond_br_12(arch)
    port map(
      condition => fork12_outs_2,
      condition_valid => fork12_outs_2_valid,
      condition_ready => fork12_outs_2_ready,
      data => buffer16_outs,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br17_trueOut,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut => cond_br17_falseOut,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  sink11 : entity work.handshake_sink_9(arch)
    port map(
      ins => cond_br17_trueOut,
      ins_valid => cond_br17_trueOut_valid,
      ins_ready => cond_br17_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer17 : entity work.handshake_buffer_18(arch)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  cond_br18 : entity work.handshake_cond_br_13(arch)
    port map(
      condition => fork12_outs_1,
      condition_valid => fork12_outs_1_valid,
      condition_ready => fork12_outs_1_ready,
      data => buffer17_outs,
      data_valid => buffer17_outs_valid,
      data_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br18_trueOut,
      trueOut_valid => cond_br18_trueOut_valid,
      trueOut_ready => cond_br18_trueOut_ready,
      falseOut => cond_br18_falseOut,
      falseOut_valid => cond_br18_falseOut_valid,
      falseOut_ready => cond_br18_falseOut_ready
    );

  sink12 : entity work.handshake_sink_10(arch)
    port map(
      ins => cond_br18_trueOut,
      ins_valid => cond_br18_trueOut_valid,
      ins_ready => cond_br18_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => cond_br16_falseOut,
      ins_valid => cond_br16_falseOut_valid,
      ins_ready => cond_br16_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  fork13 : entity work.handshake_fork_12(arch)
    port map(
      ins => cond_br17_falseOut,
      ins_valid => cond_br17_falseOut_valid,
      ins_ready => cond_br17_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  buffer21 : entity work.handshake_buffer_19(arch)
    port map(
      ins => cond_br11_falseOut,
      ins_valid => cond_br11_falseOut_valid,
      ins_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  fork14 : entity work.handshake_fork_13(arch)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  fork15 : entity work.handshake_fork_14(arch)
    port map(
      ins => cond_br18_falseOut,
      ins_valid => cond_br18_falseOut_valid,
      ins_ready => cond_br18_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  fork16 : entity work.handshake_fork_15(arch)
    port map(
      ins_valid => cond_br12_falseOut_valid,
      ins_ready => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  constant9 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => fork16_outs_1_valid,
      ctrl_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  fork17 : entity work.handshake_fork_16(arch)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
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

  mulf4 : entity work.handshake_mulf_4(arch)
    port map(
      lhs => fork13_outs_1,
      lhs_valid => fork13_outs_1_valid,
      lhs_ready => fork13_outs_1_ready,
      rhs => fork15_outs_1,
      rhs_valid => fork15_outs_1_valid,
      rhs_ready => fork15_outs_1_ready,
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
      rhs => fork17_outs_1,
      rhs_valid => fork17_outs_1_valid,
      rhs_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpf2_result,
      result_valid => cmpf2_result_valid,
      result_ready => cmpf2_result_ready
    );

  fork18 : entity work.handshake_fork_17(arch)
    port map(
      ins => cmpf2_result,
      ins_valid => cmpf2_result_valid,
      ins_ready => cmpf2_result_ready,
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

  buffer24 : entity work.handshake_buffer_20(arch)
    port map(
      ins => cond_br15_falseOut,
      ins_valid => cond_br15_falseOut_valid,
      ins_ready => cond_br15_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  select0 : entity work.handshake_select_0(arch)
    port map(
      condition => fork18_outs_2,
      condition_valid => fork18_outs_2_valid,
      condition_ready => fork18_outs_2_ready,
      trueValue => fork14_outs_1,
      trueValue_valid => fork14_outs_1_valid,
      trueValue_ready => fork14_outs_1_ready,
      falseValue => buffer24_outs,
      falseValue_valid => buffer24_outs_valid,
      falseValue_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      result => select0_result,
      result_valid => select0_result_valid,
      result_ready => select0_result_ready
    );

  buffer23 : entity work.handshake_buffer_21(arch)
    port map(
      ins => cond_br14_falseOut,
      ins_valid => cond_br14_falseOut_valid,
      ins_ready => cond_br14_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  select1 : entity work.handshake_select_1(arch)
    port map(
      condition => fork18_outs_1,
      condition_valid => fork18_outs_1_valid,
      condition_ready => fork18_outs_1_ready,
      trueValue => buffer23_outs,
      trueValue_valid => buffer23_outs_valid,
      trueValue_ready => buffer23_outs_ready,
      falseValue => fork14_outs_0,
      falseValue_valid => fork14_outs_0_valid,
      falseValue_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

  buffer25 : entity work.handshake_buffer_22(arch)
    port map(
      ins => fork13_outs_0,
      ins_valid => fork13_outs_0_valid,
      ins_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  buffer26 : entity work.handshake_buffer_23(arch)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  select2 : entity work.handshake_select_2(arch)
    port map(
      condition => fork18_outs_0,
      condition_valid => fork18_outs_0_valid,
      condition_ready => fork18_outs_0_ready,
      trueValue => buffer25_outs,
      trueValue_valid => buffer25_outs_valid,
      trueValue_ready => buffer25_outs_ready,
      falseValue => buffer26_outs,
      falseValue_valid => buffer26_outs_valid,
      falseValue_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      result => select2_result,
      result_valid => select2_result_valid,
      result_ready => select2_result_ready
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

  buffer27 : entity work.handshake_buffer_24(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork19 : entity work.handshake_fork_18(arch)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork19_outs_1,
      lhs_valid => fork19_outs_1_valid,
      lhs_ready => fork19_outs_1_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork20 : entity work.handshake_fork_19(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs(2) => fork20_outs_2,
      outs(3) => fork20_outs_3,
      outs(4) => fork20_outs_4,
      outs(5) => fork20_outs_5,
      outs(6) => fork20_outs_6,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_valid(2) => fork20_outs_2_valid,
      outs_valid(3) => fork20_outs_3_valid,
      outs_valid(4) => fork20_outs_4_valid,
      outs_valid(5) => fork20_outs_5_valid,
      outs_valid(6) => fork20_outs_6_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready,
      outs_ready(2) => fork20_outs_2_ready,
      outs_ready(3) => fork20_outs_3_ready,
      outs_ready(4) => fork20_outs_4_ready,
      outs_ready(5) => fork20_outs_5_ready,
      outs_ready(6) => fork20_outs_6_ready
    );

  cond_br19 : entity work.handshake_cond_br_14(arch)
    port map(
      condition => fork20_outs_1,
      condition_valid => fork20_outs_1_valid,
      condition_ready => fork20_outs_1_ready,
      data => select1_result,
      data_valid => select1_result_valid,
      data_ready => select1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br19_trueOut,
      trueOut_valid => cond_br19_trueOut_valid,
      trueOut_ready => cond_br19_trueOut_ready,
      falseOut => cond_br19_falseOut,
      falseOut_valid => cond_br19_falseOut_valid,
      falseOut_ready => cond_br19_falseOut_ready
    );

  sink14 : entity work.handshake_sink_11(arch)
    port map(
      ins => cond_br19_falseOut,
      ins_valid => cond_br19_falseOut_valid,
      ins_ready => cond_br19_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br20 : entity work.handshake_cond_br_15(arch)
    port map(
      condition => fork20_outs_2,
      condition_valid => fork20_outs_2_valid,
      condition_ready => fork20_outs_2_ready,
      data => select0_result,
      data_valid => select0_result_valid,
      data_ready => select0_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br20_trueOut,
      trueOut_valid => cond_br20_trueOut_valid,
      trueOut_ready => cond_br20_trueOut_ready,
      falseOut => cond_br20_falseOut,
      falseOut_valid => cond_br20_falseOut_valid,
      falseOut_ready => cond_br20_falseOut_ready
    );

  sink15 : entity work.handshake_sink_12(arch)
    port map(
      ins => cond_br20_falseOut,
      ins_valid => cond_br20_falseOut_valid,
      ins_ready => cond_br20_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br21 : entity work.handshake_cond_br_5(arch)
    port map(
      condition => fork20_outs_0,
      condition_valid => fork20_outs_0_valid,
      condition_ready => fork20_outs_0_ready,
      data => trunci0_outs,
      data_valid => trunci0_outs_valid,
      data_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br21_trueOut,
      trueOut_valid => cond_br21_trueOut_valid,
      trueOut_ready => cond_br21_trueOut_ready,
      falseOut => cond_br21_falseOut,
      falseOut_valid => cond_br21_falseOut_valid,
      falseOut_ready => cond_br21_falseOut_ready
    );

  sink16 : entity work.handshake_sink_3(arch)
    port map(
      ins => cond_br21_falseOut,
      ins_valid => cond_br21_falseOut_valid,
      ins_ready => cond_br21_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br22 : entity work.handshake_cond_br_16(arch)
    port map(
      condition => fork20_outs_3,
      condition_valid => fork20_outs_3_valid,
      condition_ready => fork20_outs_3_ready,
      data => select2_result,
      data_valid => select2_result_valid,
      data_ready => select2_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br22_trueOut,
      trueOut_valid => cond_br22_trueOut_valid,
      trueOut_ready => cond_br22_trueOut_ready,
      falseOut => cond_br22_falseOut,
      falseOut_valid => cond_br22_falseOut_valid,
      falseOut_ready => cond_br22_falseOut_ready
    );

  sink17 : entity work.handshake_sink_13(arch)
    port map(
      ins => cond_br22_falseOut,
      ins_valid => cond_br22_falseOut_valid,
      ins_ready => cond_br22_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer22 : entity work.handshake_buffer_25(arch)
    port map(
      ins => cond_br13_falseOut,
      ins_valid => cond_br13_falseOut_valid,
      ins_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  cond_br23 : entity work.handshake_cond_br_17(arch)
    port map(
      condition => fork20_outs_4,
      condition_valid => fork20_outs_4_valid,
      condition_ready => fork20_outs_4_ready,
      data => buffer22_outs,
      data_valid => buffer22_outs_valid,
      data_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br23_trueOut,
      trueOut_valid => cond_br23_trueOut_valid,
      trueOut_ready => cond_br23_trueOut_ready,
      falseOut => cond_br23_falseOut,
      falseOut_valid => cond_br23_falseOut_valid,
      falseOut_ready => cond_br23_falseOut_ready
    );

  sink18 : entity work.handshake_sink_14(arch)
    port map(
      ins => cond_br23_falseOut,
      ins_valid => cond_br23_falseOut_valid,
      ins_ready => cond_br23_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br24 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork20_outs_5,
      condition_valid => fork20_outs_5_valid,
      condition_ready => fork20_outs_5_ready,
      data_valid => fork16_outs_0_valid,
      data_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br24_trueOut_valid,
      trueOut_ready => cond_br24_trueOut_ready,
      falseOut_valid => cond_br24_falseOut_valid,
      falseOut_ready => cond_br24_falseOut_ready
    );

  cond_br25 : entity work.handshake_cond_br_18(arch)
    port map(
      condition => fork20_outs_6,
      condition_valid => fork20_outs_6_valid,
      condition_ready => fork20_outs_6_ready,
      data => fork17_outs_0,
      data_valid => fork17_outs_0_valid,
      data_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br25_trueOut,
      trueOut_valid => cond_br25_trueOut_valid,
      trueOut_ready => cond_br25_trueOut_ready,
      falseOut => cond_br25_falseOut,
      falseOut_valid => cond_br25_falseOut_valid,
      falseOut_ready => cond_br25_falseOut_ready
    );

  sink19 : entity work.handshake_sink_15(arch)
    port map(
      ins => cond_br25_trueOut,
      ins_valid => cond_br25_trueOut_valid,
      ins_ready => cond_br25_trueOut_ready,
      clk => clk,
      rst => rst
    );

  mux5 : entity work.handshake_mux_5(arch)
    port map(
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready,
      ins(0) => cond_br3_trueOut,
      ins(1) => cond_br25_falseOut,
      ins_valid(0) => cond_br3_trueOut_valid,
      ins_valid(1) => cond_br25_falseOut_valid,
      ins_ready(0) => cond_br3_trueOut_ready,
      ins_ready(1) => cond_br25_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  control_merge3 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => cond_br4_trueOut_valid,
      ins_valid(1) => cond_br24_falseOut_valid,
      ins_ready(0) => cond_br4_trueOut_ready,
      ins_ready(1) => cond_br24_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge3_outs_valid,
      outs_ready => control_merge3_outs_ready,
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready
    );

  mux6 : entity work.handshake_mux_6(arch)
    port map(
      index => control_merge4_index,
      index_valid => control_merge4_index_valid,
      index_ready => control_merge4_index_ready,
      ins(0) => cond_br11_trueOut,
      ins(1) => mux5_outs,
      ins_valid(0) => cond_br11_trueOut_valid,
      ins_valid(1) => mux5_outs_valid,
      ins_ready(0) => cond_br11_trueOut_ready,
      ins_ready(1) => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  control_merge4 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => cond_br12_trueOut_valid,
      ins_valid(1) => control_merge3_outs_valid,
      ins_ready(0) => cond_br12_trueOut_ready,
      ins_ready(1) => control_merge3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge4_outs_valid,
      outs_ready => control_merge4_outs_ready,
      index => control_merge4_index,
      index_valid => control_merge4_index_valid,
      index_ready => control_merge4_index_ready
    );

  sink20 : entity work.handshake_sink_16(arch)
    port map(
      ins_valid => control_merge4_outs_valid,
      ins_ready => control_merge4_outs_ready,
      clk => clk,
      rst => rst
    );

end architecture;
