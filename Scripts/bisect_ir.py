#!/usr/bin/env python3
"""Binary-search which IR body lines crash the Metal GPU JIT compiler.

Strategy: Take lines [0:N] from the body, strip any lines that reference
undefined SSA values (rather than replacing refs with 0), then append
`ret void`. This keeps all included instructions semantically valid.

Usage:
    python3 bisect_ir.py <path-to-failing.ll>
    python3 bisect_ir.py <path-to-failing.ll> --verbose
"""

import subprocess, sys, tempfile, os, re

IR_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scan2d_cummax_failing.ll"
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
LLVM_DIR = os.path.expanduser("~/projects/oss/llvm")
TIMEOUT = 90

with open(IR_FILE) as f:
    lines = f.readlines()

# ── Split into header / body / footer ──────────────────────────────
body_start = None
body_end = None
for i, line in enumerate(lines):
    if body_start is None and line.strip().startswith("define "):
        body_start = i + 1  # line after "define ... {"
    if body_start and body_end is None and line.strip() == "}":
        body_end = i

if body_start is None or body_end is None:
    print("ERROR: Could not find `define ... { ... }` block")
    sys.exit(1)

header = lines[:body_start]
footer = lines[body_end:]
body = lines[body_start:body_end]

# Strip existing `ret void` from body (we always add our own)
body = [l for l in body if l.strip() != "ret void"]

print(f"Header: lines 1-{body_start}")
print(f"Body:   lines {body_start+1}-{body_end} ({len(body)} instructions)")
print(f"Footer: lines {body_end+1}-{len(lines)}")
print()


# ── Helpers ────────────────────────────────────────────────────────
_DEF_RE = re.compile(r'^\s+(%\d+)\s*=')
_REF_RE = re.compile(r'%(\d+)')
# Kernel params like %0, %1, %2 are always defined
_PARAM_RE = re.compile(r'define\s+\w+\s+@\w+\(([^)]*)\)')
param_count = 0
for line in header:
    m = _PARAM_RE.search(line)
    if m:
        param_count = m.group(1).count(',') + 1
        break
PARAM_SSAS = {f'%{i}' for i in range(param_count)}


def make_ir(body_lines):
    """Build valid IR: keep only lines whose SSA refs are all defined."""
    defined = set(PARAM_SSAS)
    kept = []

    for line in body_lines:
        # What does this line define?
        dm = _DEF_RE.match(line)
        def_name = dm.group(1) if dm else None

        # What SSA values does the RHS reference?
        if dm:
            rhs = line.split('=', 1)[1]
        else:
            rhs = line
        refs = {f'%{m}' for m in _REF_RE.findall(rhs)}

        # Keep line only if all referenced values are defined
        if refs <= defined:
            kept.append(line)
            if def_name:
                defined.add(def_name)
        elif VERBOSE:
            missing = refs - defined
            print(f"    [skip] {line.rstrip()}  (missing: {missing})")

    return ''.join(header + kept + ["  ret void\n"] + footer)


def test_ir(ir_text):
    """Returns True if pipeline succeeds, False on GPU JIT crash."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir_text)
        path = f.name
    try:
        result = subprocess.run(
            ["swift", "test", "--filter", "testExternalLLIR"],
            cwd=LLVM_DIR,
            env={**os.environ, "TEST_LLIR": path},
            capture_output=True, text=True, timeout=TIMEOUT
        )
        out = result.stdout + result.stderr
        if "Test Suite 'Selected tests' passed" in out:
            return True
        if any(k in out for k in [
            "XPC_ERROR_CONNECTION_INTERRUPTED", "Code=2",
            "Code=3", "materializeAll", "Code=1"
        ]):
            return False
        # Unknown failure — treat as crash
        if VERBOSE:
            print(f"    [unknown result] exit={result.returncode}")
        return False
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


# ── Sanity checks ─────────────────────────────────────────────────
print("Testing empty body (just ret void)...")
ok = test_ir(''.join(header + ["  ret void\n"] + footer))
print(f"  Empty body: {'PASS' if ok else 'FAIL'}")
if not ok:
    print("ERROR: Even empty body crashes! Problem is in header/metadata/declares.")
    sys.exit(1)

print("Testing full body...")
full_ir = make_ir(body)
ok = test_ir(full_ir)
print(f"  Full body:  {'PASS' if ok else 'FAIL'}")
if ok:
    print("Full body passes! Nothing to bisect.")
    sys.exit(0)


# ── Binary search ─────────────────────────────────────────────────
lo, hi = 0, len(body)
print(f"\nBisecting {len(body)} body lines...")

while hi - lo > 1:
    mid = (lo + hi) // 2
    ir = make_ir(body[:mid])
    ok = test_ir(ir)
    status = "PASS" if ok else "FAIL"
    print(f"  body[:{mid:3d}] → {status}   (file line {body_start + mid})")
    if ok:
        lo = mid
    else:
        hi = mid

print(f"\n{'='*60}")
print(f"CRASH introduced at body line {hi} (file line {body_start + hi}):")
print(f"  {body[hi-1].rstrip()}")
print(f"\nContext:")
for i in range(max(0, hi - 5), min(len(body), hi + 3)):
    marker = ">>>" if i == hi - 1 else "   "
    print(f"  {marker} {body_start + i + 1:3d}: {body[i].rstrip()}")

# ── Save the minimal failing IR ───────────────────────────────────
minimal = make_ir(body[:hi])
out_path = IR_FILE.replace('.ll', '_minimal.ll')
with open(out_path, 'w') as f:
    f.write(minimal)
print(f"\nMinimal failing IR saved to: {out_path}")
