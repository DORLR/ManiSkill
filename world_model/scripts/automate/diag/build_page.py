"""Results page of milestone 1b (AutoMate parts check): one self-contained HTML (figures embedded).
Inputs: a local copy of rs2:/mnt/storage3/qingxin/automate/m1b/diag (table_m1b.json, sim/, sim/rerun/, mesh/, figs/)
and the hand-written blocks in diag/page_text.py (causes, evidence, options).
    python build_page.py <diag copy> <out.html>
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import page_text as T  # noqa: E402

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1f;--mute:#5c5c66;--line:#e2e0da;--card:#fff;--acc:#2f5d9e;--ok:#1f7a3d;--bad:#b3261e;--badbg:#fbeceb;--sh:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#ececf0;--mute:#a5a5b0;--line:#33333a;--card:#1f1f24;--acc:#8ab4f8;--ok:#47cd89;--bad:#ff8a80;--badbg:#3a2120;--sh:#3a3320}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#ececf0;--mute:#a5a5b0;--line:#33333a;--card:#1f1f24;--acc:#8ab4f8;--ok:#47cd89;--bad:#ff8a80;--badbg:#3a2120;--sh:#3a3320}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:26px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}
.note{color:var(--mute);font-size:13px}ul{padding-left:22px}li{margin:3px 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}td,th{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:600}
.wide{overflow-x:auto}.wide table{font-size:12.5px;white-space:nowrap}
tr.bad td{background:var(--badbg)} td.bad{color:var(--bad)} td.sh{background:var(--sh)}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
.opt li{margin:5px 0}
code{font-size:13px}
.k{font-weight:600}
button{font:inherit;font-size:13px;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
"""


def img(diag: Path, name: str, alt: str) -> str:
    f = diag / "figs" / name
    if not f.exists():
        return f"<p class='note'>(figure {html.escape(name)} missing)</p>"
    b = base64.b64encode(f.read_bytes()).decode()
    return f"<div class='fig'><img alt='{html.escape(alt)}' src='data:image/png;base64,{b}'></div>"


def main():
    diag, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = subprocess.run([sys.executable, str(HERE / "build_table_html.py"), str(diag / "table_m1b.json")],
                          capture_output=True, text=True, check=True).stdout
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>AutoMate Parts Check</title>",
             "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'>",
             "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'></script>",
             "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' "
             "onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\\\(',right:'\\\\)',display:false}]})\"></script>",
             f"<style>{CSS}</style></head><body><main>"]
    parts.append(T.HEADER)
    parts.append(T.SUMMARY)
    parts.append(T.CHECKS)
    parts.append(T.TABLE_INTRO)
    parts.append("<p><button onclick=\"document.querySelectorAll('#t tbody tr:not(.bad)').forEach(r=>r.hidden=!r.hidden)\">"
                 "show / hide the passing assemblies</button></p>")
    parts.append("<div class='wide'><table id='t'><thead><tr><th>id</th><th>shape</th><th>n / P</th><th>c mm</th><th>L mm</th>"
                 "<th>d_dis mm</th><th>depth check</th><th>N_max</th><th>Δθ deg</th><th>Δp mm</th><th>Δψ deg</th>"
                 "<th>(i) depth / pen mm</th><th>(ii) depth mm</th><th>(iii)</th></tr></thead><tbody>")
    parts.append(rows)
    parts.append("</tbody></table></div>")
    parts.append(T.TABLE_NOTES)
    parts.append("<h2>Failing assemblies: cause, evidence, options</h2>")
    parts.append(T.FAIL_INTRO)
    for b in T.BLOCKS:
        parts.append(f"<h3 id='a{b['id']}'>{b['id']} <span class='note'>{html.escape(b['short'])}</span></h3>")
        parts.append(b["body"])
        for name, alt in b.get("figs", []):
            parts.append(img(diag, name, alt))
        parts.append("<p class='k'>Options (for you to decide)</p><ul class='opt'>" +
                     "".join(f"<li>{o}</li>" for o in b["options"]) + "</ul>")
    parts.append(T.III)
    parts.append(T.BUGS)
    parts.append(T.FILES)
    parts.append("</main></body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
