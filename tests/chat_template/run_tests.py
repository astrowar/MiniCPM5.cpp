#!/usr/bin/env python3
"""
Validator: compares C++ chat_template::Renderer output against
the official Jinja2 chat template, case by case.

Usage:
    python run_tests.py [--build] [--cases N,M,...]

    --build       Build the C++ binary first (cmake + make)
    --cases       Comma-separated list of case numbers to run (default: all)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from jinja2 import Environment, BaseLoader
except ImportError:
    print("ERROR: jinja2 is required. Install with: pip install jinja2")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEMPLATE_PATH = HERE / "chat_template.jinja"
BUILD_DIR = Path(__file__).resolve().parent / "build"
CPP_BIN = BUILD_DIR / "Release" / ("chat_template_test.exe" if os.name == "nt" else "chat_template_test")

CASE_START = "===CASE_START==="
CASE_END = "===CASE_END==="

# ---------------------------------------------------------------------------
# Load & compile the Jinja template
# ---------------------------------------------------------------------------
def load_template():
    tpl_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    from jinja2 import ChainableUndefined
    env = Environment(loader=BaseLoader(), undefined=ChainableUndefined)

    # Patch tojson to support ensure_ascii kwarg (jinja2 <3.2 lacks it)
    import json as _json
    def _tojson(value, ensure_ascii=True, indent=None, **kwargs):
        return _json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, **kwargs)
    env.filters['tojson'] = _tojson

    return env.from_string(tpl_text)


# ---------------------------------------------------------------------------
# Test case definitions (Jinja-side representation)
# Each case is a dict with:
#   name: human-readable label
#   context: kwargs passed to the template
# ---------------------------------------------------------------------------
def build_cases():
    cases = []

    # Case 1: Simple user message
    cases.append({
        "name": "Simple user message",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "add_generation_prompt": False,
        },
    })

    # Case 2: System + user
    cases.append({
        "name": "System + user",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "add_generation_prompt": False,
        },
    })

    # Case 3: Assistant content WITHOUT think tags (renderer should insert empty block)
    cases.append({
        "name": "Assistant no-think (empty block inserted)",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "Solve 1+1"},
                {"role": "assistant", "content": "Let me think... The answer is 2."},
            ],
            "add_generation_prompt": False,
        },
    })

    # Case 4: Assistant with explicit reasoning_content
    cases.append({
        "name": "Assistant with reasoning_content",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "Solve 2+3"},
                {
                    "role": "assistant",
                    "content": "The answer is 5.",
                    "reasoning_content": "2+3 is basic arithmetic.\nI need to add them.",
                },
            ],
            "add_generation_prompt": False,
        },
    })

    # Case 5: Generation prompt, enable_thinking = True
    cases.append({
        "name": "Gen prompt, thinking=true",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    })

    # Case 6: Generation prompt, enable_thinking = False
    cases.append({
        "name": "Gen prompt, thinking=false",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    })

    # Case 7: Generation prompt, enable_thinking = undefined
    cases.append({
        "name": "Gen prompt, thinking=undefined",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "Tell me a joke"},
            ],
            "add_generation_prompt": True,
            # enable_thinking intentionally NOT set
        },
    })

    # Case 8: Tool calling - assistant + tool response + gen prompt
    cases.append({
        "name": "Tool call (weather) + response + gen",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ],
                },
                {"role": "tool", "content": '{"temp": 22, "condition": "sunny"}'},
            ],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    })

    # Case 9: Tool call with CDATA-worthy content
    cases.append({
        "name": "Tool call with CDATA (special chars)",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "Search for comparison code"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_code",
                                "arguments": {"query": "if (a < b && c > d)\n  return true;"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "Found 3 matches."},
            ],
            "add_generation_prompt": False,
        },
    })

    # Case 10: Full: system+tools+user+assistant(reasoning+tool)+tool+gen
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    cases.append({
        "name": "Full: system+tools+reasoning+tool+gen(think=false)",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "system", "content": "You are a weather assistant."},
                {"role": "user", "content": "Weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "User wants Tokyo weather. I should call get_weather.",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
                    ],
                },
                {"role": "tool", "content": '{"temp": 15, "condition": "rainy"}'},
            ],
            "tools": tools,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    })

    # Case 11: Multi-turn with thinking
    cases.append({
        "name": "Multi-turn with reasoning",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {
                    "role": "assistant",
                    "content": "AI is the simulation of intelligence by machines.",
                    "reasoning_content": "This is a broad question.\nI should give a concise definition.",
                },
                {"role": "user", "content": "Can you elaborate?"},
                {
                    "role": "assistant",
                    "content": "There are many subfields: NLP, computer vision, robotics.",
                    "reasoning_content": "The user wants more detail.\nLet me list the main branches.",
                },
            ],
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    })

    # Case 12: System with tool_def_sep
    tools2 = [
        {"name": "calc", "description": "Calculator", "parameters": {"type": "object", "properties": {}}}
    ]
    cases.append({
        "name": "System with tool_def_sep",
        "context": {
            "bos_token": "<s>",
            "messages": [
                {"role": "system", "content": "You are helpful.<tool_def_sep>Additional context here."},
                {"role": "user", "content": "2 * 3 = ?"},
            ],
            "tools": tools2,
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
    })

    return cases


# ---------------------------------------------------------------------------
# Jinja rendering
# ---------------------------------------------------------------------------
def render_jinja(template, context):
    """Render using the official Jinja template."""
    return template.render(**context)


# ---------------------------------------------------------------------------
# C++ binary execution
# ---------------------------------------------------------------------------
def run_cpp_binary():
    """Run the C++ test binary and return list of rendered outputs."""
    if not CPP_BIN.exists():
        raise FileNotFoundError(f"C++ binary not found: {CPP_BIN}\nRun with --build first.")
    result = subprocess.run(
        [str(CPP_BIN)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ binary failed (rc={result.returncode}):\n{result.stderr}")

    output = result.stdout
    # Parse between markers
    pattern = re.compile(
        re.escape(CASE_START) + r"\n(.*?)" + re.escape(CASE_END),
        re.DOTALL,
    )
    matches = pattern.findall(output)
    # The C++ printf adds a trailing \n before CASE_END; strip it
    return [m[:-1] if m.endswith('\n') else m for m in matches]


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------
def diff_strings(a: str, b: str, label: str) -> bool:
    """Print a unified-style diff if strings differ. Returns True if match."""
    if a == b:
        return True

    print(f"\n  [DIFF] {label}")
    # Show first difference
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            ctx_start = max(0, i - 20)
            ctx_end_a = min(len(a), i + 40)
            ctx_end_b = min(len(b), i + 40)
            print(f"  First diff at char {i}:")
            print(f"    Jinja: ...{repr(a[ctx_start:ctx_end_a])}...")
            print(f"    C++  : ...{repr(b[ctx_start:ctx_end_b])}...")
            break
    else:
        # One is prefix of the other
        longer = "Jinja" if len(a) > len(b) else "C++"
        shorter_len = min(len(a), len(b))
        extra = a[shorter_len:shorter_len+60] if len(a) > len(b) else b[shorter_len:shorter_len+60]
        print(f"  Length mismatch: Jinja={len(a)}, C++={len(b)}")
        print(f"  Extra in {longer}: {repr(extra)}...")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate C++ chat template vs Jinja2")
    parser.add_argument("--build", action="store_true", help="Build C++ binary first")
    parser.add_argument("--cases", type=str, default="", help="Comma-separated case numbers (1-based)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full output for each case")
    args = parser.parse_args()

    if args.build:
        print("[*] Building C++ test binary...")
        BUILD_DIR.mkdir(exist_ok=True)
        cmake_cmd = ["cmake", ".."]
        subprocess.run(cmake_cmd, cwd=BUILD_DIR, capture_output=True, check=True)
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
        subprocess.run(build_cmd, cwd=BUILD_DIR, capture_output=True, check=True)
        print(f"    Built: {CPP_BIN}")

    # Load template
    print(f"[*] Loading Jinja template: {TEMPLATE_PATH.name}")
    template = load_template()

    # Build cases
    cases = build_cases()

    # Filter if requested
    if args.cases:
        selected = set(int(x) for x in args.cases.split(","))
        cases = [(i, c) for i, c in enumerate(cases, 1) if i in selected]
    else:
        cases = list(enumerate(cases, 1))

    print(f"[*] Running {len(cases)} test case(s)...\n")

    # Run C++ binary (once, get all outputs)
    cpp_outputs = run_cpp_binary()
    if len(cpp_outputs) < len(cases):
        print(f"WARNING: C++ produced {len(cpp_outputs)} outputs, expected {len(cases)}")

    # Compare
    passed = 0
    failed = 0
    cpp_idx = 0

    for case_num, case in cases:
        ctx = case["context"]
        jinja_out = render_jinja(template, ctx)

        cpp_out = cpp_outputs[cpp_idx] if cpp_idx < len(cpp_outputs) else ""
        cpp_idx += 1

        match = diff_strings(jinja_out, cpp_out, case["name"])
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] Case {case_num:2d}: {case['name']}")

        if args.verbose and not match:
            print(f"\n  --- Jinja output (len={len(jinja_out)}) ---")
            print(jinja_out)
            print(f"\n  --- C++ output (len={len(cpp_out)}) ---")
            print(cpp_out)
            print()

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
