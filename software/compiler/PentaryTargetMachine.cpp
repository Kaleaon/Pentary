// PentaryTargetMachine.cpp - Pentary Target Machine Implementation
//
// This file implements the Pentary target machine for LLVM.
//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"

using namespace llvm;

namespace {

// Pentary Target Machine
class PentaryTargetMachine : public LLVMTargetMachine {
public:
  PentaryTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                       CodeGenOpt::Level OL, bool JIT)
      : LLVMTargetMachine(T, "e-m:e-p:48:48-i48:48-n48-S48", TT, CPU, FS,
                          Options, getEffectiveRelocModel(RM),
                          getEffectiveCodeModel(CM), OL) {
    initAsmInfo();
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

private:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
};

// Pentary Pass Config
class PentaryPassConfig : public TargetPassConfig {
public:
  PentaryPassConfig(PentaryTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  bool addInstSelector() override {
    // Add instruction selection pass
    addPass(createPentaryISelDag(getPentaryTargetMachine()));
    return false;
  }

  void addPreEmitPass() override {
    // Add passes before emission (e.g., branch relaxation)
    addPass(createPentaryBranchRelaxationPass());
  }

private:
  PentaryTargetMachine &getPentaryTargetMachine() const {
    return getTM<PentaryTargetMachine>();
  }
};

} // end anonymous namespace

TargetPassConfig *PentaryTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PentaryPassConfig(*this, PM);
}

// Register the target
extern "C" void LLVMInitializePentaryTargetInfo() {
  RegisterTarget<Triple::pentary> X(getThePentaryTarget(), "pentary",
                                     "Pentary (balanced quinary)", "Pentary");
}

extern "C" void LLVMInitializePentaryTarget() {
  // Register the target machine
  RegisterTargetMachine<PentaryTargetMachine> X(getThePentaryTarget());
}

// Target description
static Target ThePentaryTarget;

Target &getThePentaryTarget() {
  return ThePentaryTarget;
}
