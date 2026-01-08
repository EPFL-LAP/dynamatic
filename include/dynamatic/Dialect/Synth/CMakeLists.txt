##===----------------------------------------------------------------------===//
## All the files related to the description of the Synth dialect
## have been modified from the original version present in the
## circt project at the following link:
## https://github.com/llvm/circt/tree/main/
##===----------------------------------------------------------------------===//

##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_dynamatic_dialect(Synth synth)

set(LLVM_TARGET_DEFINITIONS Synth.td)