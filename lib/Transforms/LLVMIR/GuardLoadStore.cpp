// -----------------------------------------------------------------------
// Guard ALL load/stores in a function from optimizations by wrapping them
// in opaque (always-inline) function calls. 
//
// Ex. Store
// C:
//  x[idx] = var;
//
// LLVM: 
//  %gep = getelementptr inbounds i32, ptr %x, i64 %idx
//  store i32 %value, ptr %gep, align 4
//
// LLVM guarded:
//  %gep = getelementptr inbounds i32, ptr %x, i64 %idx
//  call void @__dyn_guard.store.align4.ptr.i32.to.void(ptr %gep, i32 %var)
//
// Ex. Load
// C: 
//  int var = x[idx];
//
// LLVM:
//  %gep = getelementptr inbounds i32, ptr %x, i64 %idx
//  %value = load i32, ptr %gep, align 4
//
// LLVM guarded:
//  %gep = getelementptr inbounds i32, ptr %x, i64 %idx
//  %value.g = call i32 @__dyn_guard.load.align4.ptr.to.i32(ptr %gep)
//
// NOTE:: GEP operations are unchanged
// -----------------------------------------------------------------------
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <vector>

#define DEBUG_TYPE "guard-load-store"

using namespace llvm;

namespace {

using OpaqueGuardBodyFn =
    llvm::function_ref<Value *(IRBuilder<> &, ArrayRef<Value *>)>;

// NOTE: keep in sync with whatever you already use elsewhere for naming
// guard functions distinctly per (arg types -> ret type).
std::string typeSuffix(Type *ty) {
  std::string s;
  raw_string_ostream os(s);
  ty->print(os);
  // Sanitize for use in a function name.
  for (char &c : s)
    if (!isalnum(static_cast<unsigned char>(c)))
      c = '_';
  return s;
}

CallInst *createOpaqueGuard(StringRef kind, ArrayRef<Value *> args, Type *retTy,
                            OpaqueGuardBodyFn buildBody,
                            IRBuilder<> &callSiteBuilder,
                            const Twine &callName = "") {
  Module *mod = callSiteBuilder.GetInsertBlock()->getModule();
  LLVMContext &ctx = mod->getContext();

  std::string name = ("__dyn_guard." + kind).str();
  for (Value *arg : args)
    name += "." + typeSuffix(arg->getType());
  name += ".to." + typeSuffix(retTy);

  // Lazily create the function; identical signature => identical body, so
  // reuse across call sites.
  Function *fn = mod->getFunction(name);
  if (!fn) {
    SmallVector<Type *, 4> argTypes;
    for (Value *arg : args)
      argTypes.push_back(arg->getType());

    FunctionType *fnTy = FunctionType::get(retTy, argTypes, false);
    fn = Function::Create(fnTy, GlobalValue::InternalLinkage, name, mod);
    fn->addFnAttr(Attribute::AlwaysInline);

    BasicBlock *entry = BasicBlock::Create(ctx, "entry", fn);
    IRBuilder<> b(entry);

    SmallVector<Value *, 4> fnArgs;
    fnArgs.reserve(fn->arg_size());
    for (Argument &arg : fn->args())
      fnArgs.push_back(&arg);

    Value *result = buildBody(b, fnArgs);
    retTy->isVoidTy() ? (void)b.CreateRetVoid() : (void)b.CreateRet(result);
  }

  return callSiteBuilder.CreateCall(fn, args, callName);
}

Value *guardedLoad(IRBuilder<> &b, Value *ptr, Type *elemTy, Align align,
                   const Twine &name = "") {
  std::string kind = ("load.align" + Twine(align.value())).str();
  return createOpaqueGuard(
      kind, {ptr}, elemTy,
      [elemTy, align](IRBuilder<> &body, ArrayRef<Value *> a) -> Value * {
        LoadInst *load = body.CreateLoad(elemTy, a[0]);
        load->setAlignment(align);
        return load;
      },
      b, name);
}

void guardedStore(IRBuilder<> &b, Value *ptr, Value *val, Align align) {
  std::string kind = ("store.align" + Twine(align.value())).str();
  createOpaqueGuard(
      kind, {ptr, val}, Type::getVoidTy(b.getContext()),
      [align](IRBuilder<> &body, ArrayRef<Value *> a) -> Value * {
        StoreInst *store = body.CreateStore(a[1], a[0]);
        store->setAlignment(align);
        return nullptr;
      },
      b);
}

} // namespace

struct GuardLoadStore : PassInfoMixin<GuardLoadStore> {
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

PreservedAnalyses GuardLoadStore::run(Function &f, FunctionAnalysisManager &) {

  if (f.getName() == "main") {
    return PreservedAnalyses::all();
  }

  // Don't perform this on produced guard functions or we get infinite recursion
  if (f.getName().starts_with("__dyn_guard.")) {
    return PreservedAnalyses::all();
  }

  // Collect all load and stores in separate lists
  std::vector<LoadInst *> loads;
  std::vector<StoreInst *> stores;
  for (auto &bb : f) {
    for (auto &inst : bb) {
      if (auto *ld = dyn_cast<LoadInst>(&inst)) {
        loads.push_back(ld);
      } else if (auto *st = dyn_cast<StoreInst>(&inst)) {
        stores.push_back(st);
      }
    }
  }

  // Possible early return in case there are no load/stores
  if (loads.empty() && stores.empty()) {
    return PreservedAnalyses::all();
  }

  for (LoadInst *load : loads) {
    IRBuilder<> b(load);
    Twine name = load->hasName() ? load->getName() : Twine("ld");
    Value *guarded = guardedLoad(b, load->getPointerOperand(), load->getType(),
                                 load->getAlign(), name + ".g");
    load->replaceAllUsesWith(guarded);
    load->eraseFromParent();
  }

  for (StoreInst *store : stores) {
    IRBuilder<> b(store);
    guardedStore(b, store->getPointerOperand(), store->getValueOperand(),
                 store->getAlign());
    store->eraseFromParent();
  }

  PreservedAnalyses pa;
  pa.preserveSet<CFGAnalyses>();
  return pa;
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "GuardLoadStore", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "guard-load-store") {
                    fpm.addPass(GuardLoadStore());
                    return true;
                  }
                  return false;
                });
          }};
}
