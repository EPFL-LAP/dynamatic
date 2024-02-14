set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(export-vhdl
  export-vhdl.cpp
  src/DOTParser.cpp
  src/LSQGenerator.cpp
  src/StringUtils.cpp
  src/VHDLWriter.cpp
)

llvm_update_compile_flags(export-vhdl)
