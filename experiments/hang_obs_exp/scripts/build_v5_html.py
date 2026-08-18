"""Render the v5 overnight results into a single self-contained HTML page.

Images are inlined as data URIs and videos are converted to inline base64 mp4,
because the page is published as an Artifact where a strict CSP blocks every
external request. Anything missing is rendered as an explicit gap rather than
omitted — a run where one mode died should say so on the page.

    python build_v5_html.py --report_dir logs/hang_obs_exp/v5_report \
        --out /tmp/.../v5_report.html
"""
import argparse
import base64
import json
import os


def data_uri(path, mime):
    try:
        with open(path, 'rb') as f:
            return f'data:{mime};base64,' + base64.b64encode(f.read()).decode()
    except Exception:
        return None


def img_tag(path, cap):
    u = data_uri(path, 'image/png')
    if not u:
        return f'<p class="miss">missing figure: {os.path.basename(str(path))}</p>'
    return (f'<figure><img src="{u}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


def vid_tag(path, cap, max_mb=12):
    try:
        if os.path.getsize(path) > max_mb * 1e6:
            return (f'<p class="miss">video too large to inline '
                    f'({os.path.getsize(path)/1e6:.0f} MB): <code>{path}</code></p>')
    except OSError:
        return f'<p class="miss">missing video: <code>{path}</code></p>'
    u = data_uri(path, 'video/mp4')
    if not u:
        return f'<p class="miss">missing video: <code>{path}</code></p>'
    return (f'<figure><video controls muted playsinline src="{u}"></video>'
            f'<figcaption>{cap}</figcaption></figure>')


def fmt(v, nd=3):
    if v is None:
        return '<span class="miss">—</span>'
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return str(v)


CSS = """
:root{--bg:#ffffff;--fg:#1a1a1a;--mut:#5b6472;--line:#e3e6ea;--accent:#1f6feb;
      --warn:#b45309;--card:#f7f8fa;}
:root:not([data-theme="light"]) @media (prefers-color-scheme: dark){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1216;--fg:#e6e9ee;--mut:#9aa4b2;--line:#252b33;--accent:#5a9bff;
  --warn:#f0b357;--card:#161b22;}}
:root[data-theme="dark"]{--bg:#0f1216;--fg:#e6e9ee;--mut:#9aa4b2;--line:#252b33;
  --accent:#5a9bff;--warn:#f0b357;--card:#161b22;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.2rem 1.2rem 5rem;
 font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
main{max-width:1080px;margin:0 auto}
h1{font-size:1.9rem;margin:0 0 .3rem;letter-spacing:-.02em}
h2{font-size:1.25rem;margin:2.4rem 0 .7rem;padding-bottom:.3rem;
   border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.5rem 0 .5rem;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 1.5rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600;font-size:.82rem;text-transform:uppercase;
   letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
figure{margin:1.2rem 0}
img,video{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;
  display:block}
figcaption{color:var(--mut);font-size:.85rem;margin-top:.4rem}
.miss{color:var(--warn)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:.9rem 1.1rem;margin:1rem 0}
code{background:var(--card);padding:.1rem .35rem;border-radius:4px;font-size:.88em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
ul{padding-left:1.2rem} li{margin:.3rem 0}
.kv{color:var(--mut)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_dir', required=True)
    ap.add_argument('--notes', default='')
    ap.add_argument('--extra_figs', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    summary = {}
    sp = os.path.join(a.report_dir, 'summary.json')
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)
    pol = summary.get('policies', {})
    est = summary.get('estimator', {}).get('rows', [])
    itl = summary.get('in_the_loop', {})
    vids = summary.get('videos', {})

    H = ['<style>', CSS, '</style>', '<main>']
    H.append('<h1>v5 overnight run — goal-chained demos, four BC policies, '
             'state estimator in the loop</h1>')
    H.append('<p class="sub">DEDO <code>HangProcCloth-v1</code>. '
             'Single RTX 4090, everything sequential. '
             'Equal-wall-clock budget per policy, not trained to convergence — '
             'read the cross-mode comparison as equal-budget.</p>')

    # --- headline table
    H.append('<h2>1 · Policy success by observation mode</h2>')
    H.append('<div class="wrap"><table><tr><th>obs mode</th>'
             '<th>success (legacy)</th><th>hanging</th><th>topological</th>'
             '<th>mean reward</th><th>n</th><th>status</th></tr>')
    for m in ['state', 'pcd', 'rgb', 'mesh']:
        p = pol.get(m, {})
        ok = p.get('trained')
        H.append(f'<tr><td><b>{m}</b></td>'
                 f'<td class="num">{fmt(p.get("success_rate"))}</td>'
                 f'<td class="num">{fmt(p.get("success_hanging"))}</td>'
                 f'<td class="num">{fmt(p.get("success_topological"))}</td>'
                 f'<td class="num">{fmt(p.get("reward"),1)}</td>'
                 f'<td class="num">{p.get("n_episodes") or "—"}</td>'
                 f'<td>{"ok" if ok else "<span class=miss>did not finish</span>"}</td></tr>')
    if itl.get('ran'):
        base = pol.get('state', {}).get('success_rate')
        rel = ''
        try:
            if base:
                rel = f' ({100*float(itl["success_rate"])/float(base):.0f}% of GT)'
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        H.append(f'<tr><td><b>state + estimator</b></td>'
                 f'<td class="num">{fmt(itl.get("success_rate"))}{rel}</td>'
                 f'<td class="num">—</td><td class="num">—</td>'
                 f'<td class="num">{fmt(itl.get("reward"),1)}</td>'
                 f'<td class="num">—</td><td>ok</td></tr>')
    H.append('</table></div>')
    H.append('<div class="card">The <b>state + estimator</b> row is the same '
             'policy weights as <b>state</b>; the only difference is that the '
             'hole centroid comes from the served estimator instead of the '
             'simulator. It replaces the centroid, not the mesh — so it is '
             '"the state row, made deployable", not an estimated <b>mesh</b> '
             'row.</div>')

    H.append(img_tag(os.path.join(a.report_dir, 'v5_results.png'),
                     'Success by obs mode · estimator accuracy vs occlusion · '
                     'training curves'))

    # --- estimator
    H.append('<h2>2 · State estimator, measured on its own held-out topologies</h2>')
    if est:
        H.append('<div class="wrap"><table><tr><th>point drop</th>'
                 '<th>mean occlusion</th><th>hole-centroid err (mm)</th>'
                 '<th>median (mm)</th><th>vertex err (mm)</th>'
                 '<th>extent ratio</th></tr>')
        for r in est:
            H.append(f'<tr><td class="num">{r.get("drop_frac",0):.2f}</td>'
                     f'<td class="num">{r.get("occlusion_mean",0):.2f}</td>'
                     f'<td class="num">{r.get("hole_err_mean",0)*1000:.1f}</td>'
                     f'<td class="num">{r.get("hole_err_median",0)*1000:.1f}</td>'
                     f'<td class="num">{r.get("vert_mean",0)*1000:.1f}</td>'
                     f'<td class="num">{r.get("extent_ratio_mean",0):.3f}</td></tr>')
        H.append('</table></div>')
        H.append('<div class="card">Scored through the <b>same websocket serving '
                 'path</b> the policy loop uses, on the estimator\'s own '
                 'validation split — held-out <i>topologies</i>, since every '
                 'episode re-randomizes the procedural cloth. '
                 '<b>extent ratio</b> is the one to watch: below 1.0 means the '
                 'predicted mesh is shrunk, which biases the hole centroid '
                 'toward the cloth centre while the loss still looks fine.</div>')
    else:
        H.append('<p class="miss">estimator evaluation did not produce results</p>')

    for f in a.extra_figs:
        H.append(img_tag(f, os.path.basename(f)))

    # --- videos
    H.append('<h2>3 · Rollout videos</h2><div class="grid">')
    any_v = False
    for k, paths in vids.items():
        for p in (paths or [])[:1]:
            H.append(vid_tag(p, f'{k} — eval rollouts'))
            any_v = True
    H.append('</div>')
    if not any_v:
        H.append('<p class="miss">no rollout videos were produced</p>')

    # --- design notes
    if a.notes and os.path.exists(a.notes):
        H.append('<h2>4 · Design choices, caveats, and what to re-check</h2>')
        with open(a.notes) as f:
            txt = f.read()
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith('# '):
                continue
            if ls.startswith('## '):
                H.append(f'<h3>{ls[3:]}</h3>')
            elif ls.startswith('### '):
                H.append(f'<h3>{ls[4:]}</h3>')
            elif ls.startswith(('- ', '* ')):
                H.append(f'<li>{ls[2:]}</li>')
            elif ls:
                H.append(f'<p>{ls}</p>')

    H.append('</main>')
    with open(a.out, 'w') as f:
        f.write('\n'.join(H))
    print(f'wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
