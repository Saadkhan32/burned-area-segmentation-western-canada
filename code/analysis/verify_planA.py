#!/usr/bin/env python3
"""Independent check of the benchmark outputs (re-derived from the per-window counts) plus the extra tables of the supplement.
Recomputes every headline number directly from the per-window confusion counts in results/*_test_windows.csv (not from
summary/*.csv), compares with tableP1/P2/P3 and the PowerPoint chart data, checks the quick-look index against the results,
and writes: check3_planA_outputs.md (report), tableP6_fold_membership.csv, tableP7_spatial_separation.csv,
tableP8_validation_vs_test.csv, tableP9_by_fire_year_and_season.csv, tableP1b_class_wise.csv.
Usage: python analysis/verify_planA.py --runs drac_runs --figs figures --out checks"""
import os, sys, json, glob, argparse, math, zipfile, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarize import boot_ci, sign_flip_p, holm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures_planA import MODELS, label, family, model_order

def haversine_km(lon1, lat1, lon2, lat2):
    p = math.pi / 180; a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--figs', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = a.runs; F = a.figs; rep = []
    ok = lambda cond, msg: rep.append(('PASS' if cond else 'FAIL', msg))
    # ---------------- raw counts -> metrics
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, 'results', '*_test_windows.csv')))], ignore_index=True)
    files = glob.glob(os.path.join(R, 'results', '*_test_windows.csv')); n_runs = len([f for f in files if not os.path.basename(f).startswith('dnbr_rules')])
    ok(n_runs == 42, f'42 model-run results files present ({n_runs})')
    dnbr_file = os.path.join(R, 'results', 'dnbr_rules_test_windows.csv'); has_dnbr = os.path.exists(dnbr_file)
    if has_dnbr:   # the pre-fire dNBR baseline (eval_dnbr.py): 2 rules x 135 windows, one pre-fire scene per window
        dn = pd.read_csv(dnbr_file); cand = [os.path.join(R, 'manifest_prefire.csv'), os.path.join(os.path.dirname(R.rstrip('/')), 'data_postfire', 'manifest_prefire.csv')]
        pre = next((pd.read_csv(c) for c in cand if os.path.exists(c)), None)
        ok(pre is not None, 'pre-fire manifest found next to the results (manifest_prefire.csv)')
        ok(len(dn) == 270 and set(dn.model) == {'dNBR>0.1', 'Otsu(dNBR)'} and dn.groupby('model').window_id.nunique().eq(135).all(), f'dNBR baseline file: 2 rules x 135 windows ({len(dn)} rows)')
        if pre is not None:
            ok(len(pre) == 135 and (pre.pre_status == 'OK').all() and pre.pre_clear_fraction.min() >= 0.8, f'pre-fire manifest: 135 windows, all OK, clear fraction >= 0.8 (min {pre.pre_clear_fraction.min():.3f}); |day-of-year difference| median {pre.doy_difference_days.abs().median():.0f} d, max {pre.doy_difference_days.abs().max():.0f} d')
            ok(bool((pre.pre_boa_offset_subtracted == False).all()) and (pre.pre_blue_dn_p1 < 800).all(), f'pre-fire offset check: no scene needed the 1,000-DN subtraction (blue DN p1 max {pre.pre_blue_dn_p1.max():.0f}); catalogue flag overridden for {pre.pre_offset_note.notna().sum()} scenes')
    dup = raw.duplicated(['model', 'window_id']).sum()
    raw = raw.drop_duplicates(['model', 'window_id'], keep='last')
    man = pd.read_csv(os.path.join(R, 'manifest.csv')); man = man[man.status == 'OK']
    ok('boa_offset_subtracted' in man.columns and (man.offset_note.fillna('') != '').sum() == 3, 'manifest is the corrected one (offset columns present, 3 rebuilt windows noted)')
    ok(len(man) == 135, f'manifest has 135 OK windows ({len(man)})')
    # every window scored exactly once per model (index rules appear once per run -> duplicates were dropped)
    per_model = raw.groupby('model').window_id.nunique()
    ok(bool((per_model == 135).all()), f'every model scored on all 135 windows (min {per_model.min()}, max {per_model.max()}); duplicate (index-rule) rows dropped: {dup}')
    # recompute F1 from counts and compare with the stored F1
    f1c = 2 * raw.tp / (2 * raw.tp + raw.fp + raw.fn).replace(0, np.nan)
    ok(np.nanmax(np.abs(f1c - raw.F1)) < 1e-6, 'stored F1 equals 2tp/(2tp+fp+fn) for every row')
    ok(bool(((raw.ref_ha - (raw.tp + raw.fn) / 100).abs() < 0.06).all()), 'ref_ha = (tp+fn)/100 ha for every row')
    fire = raw[raw.ref_ha > 0].copy(); neg = raw[raw.ref_ha <= 0].copy()
    ok(fire.groupby('model').size().eq(120).all() and neg.groupby('model').size().eq(15).all(), '120 fire + 15 fire-free windows per model')
    # province balance
    pv = fire[fire.model == 'NBR<0'].province.value_counts().to_dict(); ok(pv == {'AB': 40, 'BC': 40, 'SK': 40}, f'40 fire windows per province {pv}')
    # fold integrity: every window's test fold = its province, from the run configs
    cfgs = {os.path.basename(os.path.dirname(c)): json.load(open(c)) for c in glob.glob(os.path.join(R, '*', 'config.json'))}
    ok(len(cfgs) == 42, f'42 run configs ({len(cfgs)})')
    leak = 0; prov = man.set_index('window_id').province.to_dict()
    for run, c in cfgs.items():
        held = c['fold'].split('-')[1]
        tr = set(c['train_windows']); va = set(c['validation_windows']); te = set(c['test_windows'])
        if tr & te or va & te or tr & va: leak += 1
        if any(prov[w] == held for w in tr | va) or any(prov[w] != held for w in te): leak += 1
    ok(leak == 0, 'no window shared between train / validation / test in any run; test = held-out province only')
    # ---------------- table P1 vs recomputation
    t1 = pd.read_csv(os.path.join(F, 'tableP1_summary_by_model.csv'))
    maxdiff = 0
    for _, r in t1.iterrows():
        m = next(k for k, v in MODELS.items() if v[0] == r.Model); g = fire[fire.model == m]
        lo, hi = boot_ci(g.F1)
        d = max(abs(g.F1.mean() - r['Mean F1']), abs(g.F1.median() - r['Median F1']), abs(g.P.mean() - r.Precision), abs(g.R.mean() - r.Recall),
                abs(g[g.province == 'BC'].F1.mean() - r['F1 BC']), abs(lo - float(r['95 % CI'].split('-')[0])), abs(hi - float(r['95 % CI'].split('-')[1])))
        maxdiff = max(maxdiff, d)
    ok(maxdiff < 0.0006, f'tableP1 means, medians, P, R, per-province means and bootstrap CIs reproduce from raw counts (max |diff| {maxdiff:.4f})')
    # ---------------- paired differences spot check (every model vs NBR<0 and the planned pairs)
    wide = fire.pivot(index='window_id', columns='model', values='F1'); t2 = pd.read_csv(os.path.join(F, 'tableP2_planned_comparisons.csv'))
    inv = {v[0]: k for k, v in MODELS.items()}; md = 0; n_checked = 0
    for _, r in t2.iterrows():
        if r.A in inv and r.B in inv:
            d = (wide[inv[r.A]] - wide[inv[r.B]]).dropna(); md = max(md, abs(d.mean() - r['Mean F1 difference (A - B)'])); n_checked += 1
            ok(int((d > 0).sum()) == int(r['Windows A better']), f'windows won reproduce: {r.A} vs {r.B} ({int((d > 0).sum())})')
    ok(md < 1e-4, f'tableP2 mean differences reproduce ({n_checked} pairs, max |diff| {md:.5f})')
    if os.path.exists(os.path.join(F, 'tableP2b_dnbr_baseline.csv')):
        t2b = pd.read_csv(os.path.join(F, 'tableP2b_dnbr_baseline.csv')); md = 0; n_checked = 0; wins_ok = True
        for _, r in t2b.iterrows():
            if r.A in inv and r.B in inv:
                d = (wide[inv[r.A]] - wide[inv[r.B]]).dropna(); md = max(md, abs(d.mean() - r['Mean F1 difference (A - B)'])); n_checked += 1
                wins_ok &= int((d > 0).sum()) == int(r['Windows A better'])
        ok(md < 1e-4 and wins_ok and n_checked == len(t2b), f'tableP2b (dNBR baseline) mean differences and windows won reproduce ({n_checked} pairs, max |diff| {md:.5f})')
    # ---------------- fire-free table
    t3 = pd.read_csv(os.path.join(F, 'tableP3_fire_free_windows.csv')); d3 = 0
    for _, r in t3.iterrows():
        m = inv[r.Model]; g = neg[neg.model == m]; d3 = max(d3, abs(g.pred_ha.mean() - r['Mean mapped burn (ha)']), abs(g.pred_ha.max() - r['Largest mapped burn (ha)']))
    ok(d3 < 0.51, f'tableP3 fire-free means/max reproduce (max |diff| {d3:.2f} ha)')
    # ---------------- PowerPoint chart data vs CSV
    z = zipfile.ZipFile(os.path.join(F, 'figures_planA_editable_charts.pptx')); xml = z.read('ppt/charts/chart1.xml').decode() if 'ppt/charts/chart1.xml' in z.namelist() else ''
    vals = [float(v) for v in re.findall(r'<c:v>([0-9.]+)</c:v>', xml)]
    means = set(round(float(x), 3) for x in t1['Mean F1'])
    hit = len(means & set(round(v, 3) for v in vals))
    ok(hit >= len(means), f'PowerPoint slide 1 contains every mean F1 of tableP1 ({hit} matches of {len(means)} values)')
    # ---------------- quick-look index vs results
    qi = os.path.join(R, 'quicklook', 'quicklook_index.json')
    if os.path.exists(qi):
        idx = json.load(open(qi)); mism = 0; n = 0
        for w, e in idx.items():
            for m, s in e['models'].items():
                r = raw[(raw.model == m) & (raw.window_id == w)]
                if len(r) and s['F1'] == s['F1']:
                    n += 1; mism += abs(float(r.F1.iloc[0]) - s['F1']) > 1e-6
        ok(mism == 0, f'quick-look index scores match the results ({n} entries checked)')
    # ---------------- figures exist at 600 dpi
    from PIL import Image
    for f in sorted(glob.glob(os.path.join(F, 'figP*.png'))):
        im = Image.open(f); dpi = im.info.get('dpi', (0, 0))[0]
        ok(abs(dpi - 600) < 1, f'{os.path.basename(f)}: {im.size[0]}x{im.size[1]} px at {dpi:.0f} dpi')
    # ---------------- extra tables
    # P6 fold membership
    rows = []
    for w, r in man.set_index('window_id').iterrows():
        d = dict(window_id=w, province=r.province, source=r.source, fire_year=r.fire_year, burned_ha=r.burned_ha, is_negative=r.is_negative, acquisition=str(r.acquisition)[:10], lon=round(r.lon, 4), lat=round(r.lat, 4))
        for fold in ('F-AB', 'F-BC', 'F-SK'):
            c = cfgs[f'unet_{fold}']; d[f'role_{fold}'] = 'TEST' if w in c['test_windows'] else ('VALIDATION' if w in c['validation_windows'] else 'TRAIN')
        rows.append(d)
    t6 = pd.DataFrame(rows).sort_values(['province', 'window_id']); t6.to_csv(os.path.join(F, 'tableP6_fold_membership.csv'), index=False)
    # P7 spatial separation: distance from every test window to the nearest train/validation window of the same fold
    rows = []
    for fold in ('F-AB', 'F-BC', 'F-SK'):
        te = t6[t6[f'role_{fold}'] == 'TEST']; dev = t6[t6[f'role_{fold}'] != 'TEST']
        dmin = [min(haversine_km(r.lon, r.lat, q.lon, q.lat) for _, q in dev.iterrows()) for _, r in te.iterrows()]
        rows.append({'Fold': fold, 'Test windows': len(te), 'Development windows': len(dev), 'Nearest development window to a test window (km)': round(min(dmin), 1),
                     'Median nearest distance (km)': round(float(np.median(dmin)), 1), 'Test windows within 25 km of a development window': int(sum(d < 25 for d in dmin)),
                     'Test windows within 50 km': int(sum(d < 50 for d in dmin))})
    allw = t6.reset_index(drop=True); dd = []
    for i in range(len(allw)):
        for j in range(i + 1, len(allw)):
            dd.append(haversine_km(allw.lon[i], allw.lat[i], allw.lon[j], allw.lat[j]))
    t7 = pd.DataFrame(rows); t7.to_csv(os.path.join(F, 'tableP7_spatial_separation.csv'), index=False)
    rep.append(('INFO', f'minimum centre-to-centre distance between any two windows: {min(dd):.1f} km (windows are 20 km wide)'))
    # P8 validation vs test per run
    rows = []
    for run, c in sorted(cfgs.items()):
        L = json.load(open(os.path.join(R, run, 'ledger.json'))); m = c['model'] + ('_' + c['tag'] if c.get('tag') else '')
        g = fire[(fire.model == m) & (fire.province == c['fold'].split('-')[1])]
        rows.append({'Model': label(m), 'Fold': c['fold'], 'Validation windows': len(c['validation_windows']), 'Best validation mean F1': round(L['best_val_F1_mean_window'], 3), 'Epoch of best': int(json.load(open(os.path.join(R, run, 'ledger.json'))).get('epochs_run', 0)) - 10,
                     'Test mean F1 (held-out province)': round(g.F1.mean(), 3), 'Test median F1': round(g.F1.median(), 3), 'Drop (validation - test)': round(L['best_val_F1_mean_window'] - g.F1.mean(), 3)})
    t8 = pd.DataFrame(rows); order = {label(m): i for i, m in enumerate(model_order(pd.read_csv(os.path.join(R, 'summary', 'summary_by_model.csv'))))}
    t8['o'] = t8.Model.map(order); t8 = t8.sort_values(['o', 'Fold']).drop(columns='o'); t8.to_csv(os.path.join(F, 'tableP8_validation_vs_test.csv'), index=False)
    # P9 by fire year and scene season
    mm = man.set_index('window_id'); fire['fire_year'] = fire.window_id.map(mm.fire_year); fire['season'] = np.where(fire.window_id.map(mm.scene_year_offset) > 0, 'next season', 'same season')
    rows = []
    for m in model_order(pd.read_csv(os.path.join(R, 'summary', 'summary_by_model.csv'))):
        g = fire[fire.model == m]; d = {'Model': label(m)}
        for y, gg in g.groupby('fire_year'): d[f'F1 fires of {y} (n = {len(gg)})'] = round(gg.F1.mean(), 3)
        for s, gg in g.groupby('season'): d[f'F1 {s} scenes (n = {len(gg)})'] = round(gg.F1.mean(), 3)
        rows.append(d)
    t9 = pd.DataFrame(rows); t9.to_csv(os.path.join(F, 'tableP9_by_fire_year_and_season.csv'), index=False)
    # P1b class-wise metrics (burned and unburned class, mean over windows) from the counts
    rows = []
    for m in model_order(pd.read_csv(os.path.join(R, 'summary', 'summary_by_model.csv'))):
        g = fire[fire.model == m]
        uf1 = 2 * g.tn / (2 * g.tn + g.fn + g.fp); uP = g.tn / (g.tn + g.fn); uR = g.tn / (g.tn + g.fp)
        rows.append({'Model': label(m), 'Burned F1': round(g.F1.mean(), 3), 'Burned precision': round(g.P.mean(), 3), 'Burned recall': round(g.R.mean(), 3), 'Burned IoU': round(g.IoU.mean(), 3),
                     'Unburned F1': round(uf1.mean(), 3), 'Unburned precision': round(uP.mean(), 3), 'Unburned recall': round(uR.mean(), 3), 'Pixel accuracy': round(((g.tp + g.tn) / g.valid_px).mean(), 3),
                     'Omission (ha, median)': round((g.fn / 100).median(), 0), 'Commission (ha, median)': round((g.fp / 100).median(), 0)})
    t1b = pd.DataFrame(rows); t1b.to_csv(os.path.join(F, 'tableP1b_class_wise.csv'), index=False)
    for name, t in (('tableP6_fold_membership', t6), ('tableP7_spatial_separation', t7), ('tableP8_validation_vs_test', t8), ('tableP9_by_fire_year_and_season', t9), ('tableP1b_class_wise', t1b)):
        open(os.path.join(F, name + '.md'), 'w').write(t.to_markdown(index=False) + '\n')
    # ---------------- report
    fails = sum(1 for s, _ in rep if s == 'FAIL')
    lines = [f'# Check - benchmark outputs re-derived from the per-window counts ({pd.Timestamp.now():%Y-%m-%d %H:%M})', '',
             f'Source: {R} (42 runs). Result: {fails} FAIL, {sum(1 for s, _ in rep if s == "PASS")} PASS.', '']
    lines += [f'- [{s}] {m}' for s, m in rep]
    lines += ['', '## Extra tables written to the figures folder', '- tableP6_fold_membership.csv: role of every window in each fold (release file for the split).',
              '- tableP7_spatial_separation.csv: distance from each test window to the nearest development window of its fold.',
              '- tableP8_validation_vs_test.csv: best validation F1 (development provinces) against test F1 (held-out province) per run (R3.7).',
              '- tableP9_by_fire_year_and_season.csv: mean F1 by fire year and by same-/next-season scene (R1.8 multi-year).',
              '- tableP1b_class_wise.csv: burned and unburned class metrics, pixel accuracy, omission/commission hectares (protocol secondary metrics).', '',
              '## Spatial separation', t7.to_markdown(index=False), '', '## Validation vs test (first rows)', t8.head(12).to_markdown(index=False), '', '## By fire year and season', t9.to_markdown(index=False)]
    open(os.path.join(a.out, 'check3_planA_outputs.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(f'[{s}] {m}' for s, m in rep)); print('->', os.path.join(a.out, 'check3_planA_outputs.md'))

if __name__ == '__main__':
    main()
