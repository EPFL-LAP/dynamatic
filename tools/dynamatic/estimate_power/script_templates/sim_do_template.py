################################################################
# BASE SIMULATION DO STRUCTURE TEMPLATE
################################################################
base_simulation_do = r"""
vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
%{inputs}
project addfile %{designsrc}
project calculateorder
project compileall
eval vsim tb
power add %powerflag /duv_inst/*
log -r *
run -all
power report -all -bsaif %{stage}.saif
exit
"""