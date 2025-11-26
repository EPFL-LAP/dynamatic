################################################################
# BASE XDC TEMPLATE
################################################################
base_xdc = """
create_clock -period %tcp -name clk -waveform {0.000 %halftcp} [get_ports clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

"""
