set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(banba-tblgen BANBA
  banba-tblgen.cpp
  NamedDriversGen.cpp
  PortNamesGen.cpp
  ResultNamesGen.cpp
)

target_link_libraries(banba-tblgen
  PRIVATE
  MLIRTableGen
  MLIRTblgenLib
)

llvm_update_compile_flags(banba-tblgen)
