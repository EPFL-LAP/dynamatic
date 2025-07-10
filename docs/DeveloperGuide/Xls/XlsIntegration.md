# XLS Integration

## Overview

[XLS](https://github.com/google/xls) is an open-source, data-flow oriented HLS
tool developed by Google, with quite a potential for synergy with Dynamatic: 
In short, Dynamatic is very good at designing networks of data flow
units, while XLS is very good at synthesizing and implementing arbitrary data
flow units.

Very recently XLS gained an MLIR dialect and interface, greatly simplifying potential
inter-operability between the two.

This MLIR dialect is available in Dynamatic if enabled at compilation
(`--experimental-enable-xls` flag in `./build.sh`).

This documents serves as an overview of this integration, because due some
unfortunate points of friction, it is not quite as straight forward as one might
hope.

## Challenges

Specifically, integration is hindered by two issues:

- XLS uses the `bazel` build system, and does not rely on the standard MLIR
  CMake infrastructure. As such, it has a very different project and file structure,
  that does not cleanly integrate in Dynamatic.

- XLS is religiously updated to the newest version of LLVM, with new upstream
  versions often being pinned multiple times a day, while Dynamatic is stuck on
  the LLVM version used by Polygeist, which is more than two years out of date.

## Goals

In this light, the integration was designed with the following in mind:

- *Be opt-in*: Since XLS is quite a large dependency and the integration is 
  built on somewhat shaky ground, it is completely disabled by default. This
  hopefully prevents friction during "mainline" Dynamatic development.

- *Rely on upstream XLS as much as possible*: While it is currently impossible
  to use a "vanilla" checkout of XLS, the amount of patching of XLS code is
  kept to a minimum and done in a fashion that (hopefully) enables relatively 
  simple updating to a new version of XLS.

- *Be isolated*: Minimize the amount of toggles/conditional code paths required
  in "mainline" XLS tools like `dynamatic-opt` to handle the presence/absence of
  XLS.

## The Gory Details

### Pulling-in XLS

Since XLS is quite large, it is not included as a git submodule as this would
see it downloaded by default, even if not required.

Instead, the `build.sh` script fetches the correct version of XLS to `xls/` if
XLS integration is enabled during configuration.

While building, the `build.sh` script verifies that the `xls/` checkout is
at the correct commit/version. If this is not the case, it will print a warning
message but it *will not* automatically update to the correct version to avoid
deleting work.

The upstream XLS git URL and commit hash are set in `build.sh`. Note that we
use a fork[^2] of XLS with minimal compatibility changes. See [below](#overcoming-llvm-version-differences).

### Conditional Inclusion

If XLS is enabled, `build.sh` sets the CMake variable `DYNAMATIC_ENABLE_XLS`
which is in turn is used to enable XLS-specific libraries and targets. This also 
causes the `DYNAMATIC_ENABLE_XLS` macro to be defined for all C++ and Tablegen
targets to allow for conditional compilation.

### General Structure

XLS-specific passes were simply added to normal XLS pass sets (like `Conversion/Passes.td`, 
or `experimental`'s `Transforms/Passes.td`) and gates using `DYNAMATIC_ENABLE_XLS`,
this will still require all dynamatic tools and libraries that uses these passes
to link against the XLS dialect if `DYNAMATIC_ENABLE_XLS` is set. While the dialect
is not particularly large, this would `CMakeLists.txt` all over Dynamatic.

Instead, all XLS-specific passes, dialects, and code is placed in its own
folder hierarchy (located at `experimental/xls`), featuring its own `include`,
folder, Pass sets, and namespace (`dynamatic::experimental::xls`). 

With this setup, only tools that explicitly require XLS features and import
headers from this hierarchy need to link against the XLS dialect and passes
when `DYNAMATIC_ENABLE_XLS` is set.

This subsystem also features a dedicated test suite that can be run
using `ninja check-dynamatic-xls`.

### Overcoming LLVM version differences

Just like any other dialect, the XLS MLIR dialect consists of Tablegen
definition (the "ODS") and C++ source files. Both are naturally
written against the up-to-date version of LLVM used by XLS.

To enable translation, we require at least one binary that includes both the
Handshake and XLS dialect specification. Because it lives in the Dynamatic
repo, this integration takes the route of back-porting the MLIR dialect to the
version of LLVM used in Dynamatic.  

This means we must compile the Tablegen ODS with our 2023 version of
`mlir-tblgen`, which does not work out of the box due to small changes in the
ODS structure of the years. For example, the XLS ODS triggers an `mlir-tblgen`
bug that is fixed upstream but not available in our version [^1].

Similarly, we need to compile and link the dialect source files against our 
version of LLVM, which features slightly different APIs.

To overcome this, we use a fork[^2] of XLS with a small set of patches that
work around these differences conditionally if the `DYNAMATIC_ENABLE_XLS` macro
is present.

For example, in Dynamatic's LLVM version, `LogicalResult` lives in `mlir/`, while
in upstream LLVM it has been moved to `llvm/`:

```cpp
#ifdef DYNAMATIC_ENABLE_XLS
// Header name changed in LLVM
#include "mlir/include/mlir/Support/LogicalResult.h"
#else
#include "llvm/include/llvm/Support/LogicalResult.h"
#endif  // DYNAMATIC_ENABLE_XLS
```

The conditionally inclusion of all these fixes keeps the patched version compatible
with XLS, allowing the correct version of `XLS` to be built inside
`xls/` if desired.

It is suprising how few changes are needed to get this to compile and pass a first
smoke test, given that there are 50'000+ commits between the two LLVM versions.
Still, *this is not a good and permanent solution*. There is a very high 
likelyhood that there are subtle (or even not so subtle) changes in behaviour that
do not prevent the dialect from compiling change its semantics.

## Notes

### Updating XLS

To pin a new version of XLS, the steps are *roughly* as follows:

- Pull new XLS commits from upstream to the `main` of the XLS fork[^2].
- Check-out the XLS commit which you wish to pin:
  ```bash
  git checkout <HASH>
  ```
- Create a new `dynamatic_interop` branch at this commit
  ```bash
  git checkout -b "dynamatic_interop_$(date '+%Y_%m_%d')"
  ```
- Re-apply the patches from the previous `dynamatic_interop` branch on your new branch:
  ```bash
  git cherry-pick <HASH OF PREVIOUS PATCH COMMIT>
  ```
  Note that you potentially have to update the patches to be compatible with the
  new version of XLS.
- Validate that the XLS+Dynamatic integration works with this new version and
  patch set.
- Push the new `dynamatic_interop` branch to our fork.
- Update `XLS_COMMIT` in `build.sh` to the hash of last commit if your new branch.

Note that we intend to *keep* the previous integration branches and patch sets
around (hence the new branch with date). This ensures that the XLS version and patch
set combination relied on by older versions of dynamatic remain available.

[^1]: https://github.com/llvm/llvm-project/pull/122717 
[^2]: https://github.com/ETHZ-DYNAMO/xls 
