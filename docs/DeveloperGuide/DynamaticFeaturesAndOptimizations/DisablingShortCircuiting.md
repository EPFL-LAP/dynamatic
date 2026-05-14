# Disabling short-circuit evaluation

> [!NOTE]
> Rough draft — rationale and prose to be rewritten by the author. The
> structure and bullet points below sketch the argument; the final text
> belongs to the author.

## What this is

A clang plugin at `tools/clang-plugins/no-short-circuit/` that rewrites
every `a && b` in the input C source to `(!!(a)) & (!!(b))` and every
`a || b` to `(!!(a)) | (!!(b))`. The plugin runs as the first step of
`compile.sh` and produces a rewritten `.c` file that is consumed by the
rest of the frontend pipeline. It is on by default; the
`--enable-short-circuiting` flag on `dynamatic` disables it.

## Why

Short-circuit `&&` and `||` lower to a CFG diamond in LLVM IR: the
right-hand operand is gated by a conditional branch on the left-hand
result, so that the RHS is skipped when the result is already determined
by the LHS. In a dataflow circuit, that diamond materialises as control
branches, control merges, and the buffering they need. The compound
boolean expression becomes a small control sub-graph instead of a single
arithmetic computation.

The bitwise rewrite removes the diamond. `(!!a) & (!!b)` and
`(!!a) | (!!b)` are pure arithmetic: both operands are always evaluated,
the `!!` idiom coerces each to a 0/1 boolean, and the bitwise operator
produces the same boolean result. No control flow is introduced.

The two forms are not generally equivalent in C: with side-effecting
operands the short-circuit form skips the side effect on the
short-circuited side, and the rewritten form does not. They are
equivalent when both operands are pure expressions, which is the case
for typical synthesisable HLS kernels (no calls, no assignments inside
conditionals).

## Default-on, opt-out

The plugin is enabled by default because the rewrite is a clear win for
the kernels Dynamatic targets and the semantic change is harmless when
operands are pure. The `--enable-short-circuiting` flag exists for code
that intentionally relies on short-circuit semantics — most commonly a
guard-and-deref pattern such as `p != NULL && *p > 0`, where evaluating
the RHS unconditionally would be a bug.

When the plugin rewrites at least one operator in a translation unit it
emits a single clang warning naming the input file and the number of
rewrites, so the user is aware that the source has been transformed.

## Implementation outline

- Clang `PluginASTAction` (`AddAfterMainAction`) registered as
  `noshortcircuit`. Built as `NoShortCircuitPlugin.so` via the
  `add_llvm_library(... PLUGIN_TOOL clang)` pattern (matches the
  `speculate` plugin; avoids `cl::opt` double-registration).
- `RecursiveASTVisitor::VisitBinaryOperator` filters `BO_LAnd` /
  `BO_LOr`. Operators whose location is not in the main file are
  skipped, so macros and headers are not rewritten. Operand ranges are
  resolved through `getExpansionLoc` and the rewrite is skipped if the
  resolved range does not land cleanly in the main file.
- Each kept operator gets `(!!( ... ))` wrapped around each operand and
  its 2-character token replaced with `& ` or `| ` (preserving column
  offsets for the rest of the line).
- A single warning is emitted at the start of the main file when the
  rewrite count is nonzero.
- Plugin argument `out=<path>` is required and names the file to which
  the (rewritten or original) main-file buffer is written.

## CLI plumbing

- `dynamatic --enable-short-circuiting` sets a flag that is forwarded
  to `compile.sh` as a positional argument.
- `compile.sh` always produces a file at `$F_C_REWRITTEN` so the rest
  of the pipeline consumes the same path regardless of the flag.
  - Default (flag not set): clang runs with `-fplugin=...` and the
    plugin writes the rewritten buffer to `$F_C_REWRITTEN`.
  - With `--enable-short-circuiting`: clang plugin is skipped and the
    script copies the original `.c` to `$F_C_REWRITTEN`.
