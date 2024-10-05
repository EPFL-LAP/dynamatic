# Simulating floating-point AMD/Xilinx libraries with ModelSim

The following guidelines have been extracted from the reference documentation that you can find at the following link:

https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Simulating-with-Third-Party-Simulators

### Prerequisites

In order to complete the procedure, make sure you have **compatible versions** of Vivado and ModelSim. The following links contain lists of compatible versions:

https://www.xilinx.com/support/answers/68324.html
https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Compatible-Third-Party-Tools

### Compile the libraries from Vivado

The simulation libraries need to be extracted from Vivado and compiled for the specific version of modelsim installed. To achieve this, the following steps are required.

In Vivado, select *Tools -> Compile simulation libraries -> ModelSim simulator*, and set the path to where your ModelSim is installed. Make sure to set the path for compiled libraries, otherwise they will be written to your home folder.

Select then all the libraries and confirm the generation. The process might take a long time, from several minutes up to an hour.

### Include libraries into ModelSim project
Once the export is done, the simulation libraries should be included into your simulation environment.
To do this, the file **modelsim.ini** should be edited in order contain the names of the floating point libraries as well, for instance:

``floating_point_v7_1_4 = ../modelsim_lib/floating_point_v7_1_4``

The modelsim.ini file present is in your ModelSim project.

You can now run the simulation libraries within the simulation environment.

### FAQ

**Use VHDL 2008**: use this option to compile the project files.

**Time resolution**: in the modelsimproject tab, choose *add to project -> simulation configuration -> select top module -> set resolution to 'ps'*.
