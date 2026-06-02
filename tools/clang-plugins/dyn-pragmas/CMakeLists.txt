add_llvm_library(DynPragmasPlugin MODULE
  Plugin.cpp
  PLUGIN_TOOL clang
)

# If building LLVM from source
# LLVM must be built before the plugin
if(DYNAMATIC_BUILD_LLVM)
  add_dependencies(DynPragmasPlugin clang-tablegen-targets)
endif()
