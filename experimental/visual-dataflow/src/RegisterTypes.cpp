#include "RegisterTypes.h"
#include "VisualDataflow.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initializeModule(ModuleInitializationLevel level) {
  if (level != MODULE_INITIALIZATION_LEVEL_SCENE) {
    return;
  }

  ClassDB::register_class<VisualDataflow>();
}

void terminateModule(ModuleInitializationLevel level) {
  if (level != MODULE_INITIALIZATION_LEVEL_SCENE) {
    return;
  }
}

extern "C" {

GDExtensionBool GDE_EXPORT
visualDataflowInit(GDExtensionInterfaceGetProcAddress getProdAddress,
                   GDExtensionClassLibraryPtr library,
                   GDExtensionInitialization *initialization) {
  godot::GDExtensionBinding::InitObject initObj(getProdAddress, library,
                                                initialization);

  initObj.register_initializer(initializeModule);
  initObj.register_terminator(terminateModule);
  initObj.set_minimum_library_initialization_level(
      MODULE_INITIALIZATION_LEVEL_SCENE);

  return initObj.init();
}
}
