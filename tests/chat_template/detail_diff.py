"""Show exact byte-level difference for case 8."""
import json, subprocess, re
from pathlib import Path
from jinja2 import Environment, BaseLoader, ChainableUndefined

HERE = Path(__file__).resolve().parent
tpl = Environment(loader=BaseLoader(), undefined=ChainableUndefined)
tpl.filters['tojson'] = lambda v, ensure_ascii=True, **kw: json.dumps(v, ensure_ascii=ensure_ascii, **kw)
t = tpl.from_string((HERE / "chat_template.jinja").read_text(encoding='utf-8'))

jinja_out = t.render(
    bos_token='<s>',
    messages=[
        {'role':'user','content':"What's the weather in Paris?"},
        {'role':'assistant','content':'','tool_calls':[{'function':{'name':'get_weather','arguments':{'city':'Paris'}}}]},
        {'role':'tool','content':'{"temp": 22, "condition": "sunny"}'},
    ],
    add_generation_prompt=True,
    enable_thinking=True,
)

# Get C++ output
result = subprocess.run(
    [str(Path(__file__).resolve().parent / "build" / "Release" / "chat_template_test.exe")],
    capture_output=True, text=True, timeout=30,
)
pattern = re.compile(r'===CASE_START===\n(.*?)===CASE_END===', re.DOTALL)
matches = pattern.findall(result.stdout)
cpp_out = matches[7]  # case 8 (0-indexed)
if cpp_out.endswith('\n'):
    cpp_out = cpp_out[:-1]

# Find all differences
diffs = []
for i in range(min(len(jinja_out), len(cpp_out))):
    if jinja_out[i] != cpp_out[i]:
        diffs.append(i)

print(f"Jinja len: {len(jinja_out)}, C++ len: {len(cpp_out)}")
print(f"Number of char diffs: {len(diffs)}")
for i in diffs[:5]:
    ctx_j = repr(jinja_out[max(0,i-10):i+10])
    ctx_c = repr(cpp_out[max(0,i-10):i+10])
    print(f"  At {i}: Jinja={ctx_j} | C++={ctx_c}")

if len(jinja_out) != len(cpp_out):
    longer = 'Jinja' if len(jinja_out) > len(cpp_out) else 'C++'
    shorter_len = min(len(jinja_out), len(cpp_out))
    print(f"Length diff: {longer} has {abs(len(jinja_out)-len(cpp_out))} extra chars")
    print(f"  Extra: {repr((jinja_out if len(jinja_out)>shorter_len else cpp_out)[shorter_len:shorter_len+30])}")
