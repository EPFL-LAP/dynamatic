//===- MarkMemoryInterfaceProtocol.h - Mark Memory Interface Protocol ----*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --hw-mark-li-memory-interface pass
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_MARKMEMORYINTERFACEPROTOCOL_H
#define DYNAMATIC_TRANSFORMS_MARKMEMORYINTERFACEPROTOCOL_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

// include tblgen base class declaration,
// options struct
// and pass create function
#define GEN_PASS_DECL_MARKMEMORYINTERFACEPROTOCOL
#define GEN_PASS_DEF_MARKMEMORYINTERFACEPROTOCOL
#include "dynamatic/Transforms/Passes.h.inc"
std::unique_ptr<dynamatic::DynamaticPass>
createMarkMemoryInterfaceProtocolPass(std::string protocol = "synchronous");

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_MARKMEMORYINTERFACEPROTOCOL_H
