//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ResourceSharingFCCM22 pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H

#include "mlir/Pass/PassManager.h"
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

#define GEN_PASS_DECL_RESOURCESHARINGFCCM22
#define GEN_PASS_DEF_RESOURCESHARINGFCCM22
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createResourceSharingFCCM22Pass(StringRef algorithm = "fpga20",
                                StringRef frequencies = "",
                                StringRef timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H
