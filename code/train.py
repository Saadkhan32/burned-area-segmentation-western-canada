#!/usr/bin/env python3
"""Train one model on one leave-one-province-out fold.
Design: whole windows are the unit; the held-out province is never read;
three reserved development windows per fold are the validation set (whole-window sliding inference each epoch, early stopping
on their mean burned-class F1); training samples 256 x 256 crops from the remaining windows with burned-centred oversampling;
normalisation statistics and class weights come from the training windows only; CNNs share one objective
(class-weighted cross-entropy + 0.5 soft Dice); Mask2Former uses its native loss; SAM-LoRA the same CE + Dice.

Usage (one run):  python train.py --data data_postfire --fold F-AB --model unet --out runs
Outputs in runs/<model>_<fold>/: config.json, norm_stats.json, log.csv, best.pt, last.pt, ledger.json (sha256 of best.pt).
"""
import os, sys, csv, json, time, math, argparse, hashlib, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import read_manifest, fold_roles, read_window, confusion, prf, sliding_logits, sha256, IGNORE
from models import build_model, mask2former_targets, N_CLASSES

DEFAULT_VALIDATION = {   # reserved validation windows per fold (chosen on 12 Oct 2026, before any result was seen)
    'F-AB': ['Greer_Creek', 'Tatuk_Lake', 'HEINZ'],
    'F-BC': ['Dillion_River_Fire', 'Rabbit_Lake', 'COWIE'],
    'F-SK': ['Pony_Creek', 'Murtle_Lake', 'Satah_Mountain'],
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class CropSampler:
    """Holds the training windows in memory and draws augmented crops; burned-centred with probability p_pos."""
    def __init__(self, windows, crop, p_pos, rng):
        self.w = windows; self.crop = crop; self.p = p_pos; self.rng = rng
        self.pos = {}
        for wid, (img, lab) in windows.items():
            ys, xs = np.nonzero(lab == 1)
            self.pos[wid] = np.stack([ys, xs], 1) if len(ys) else None
        self.ids = list(windows)

    def draw(self):
        wid = self.ids[self.rng.integers(len(self.ids))]
        img, lab = self.w[wid]; H, W = lab.shape; c = self.crop
        if self.pos[wid] is not None and self.rng.random() < self.p:
            y, x = self.pos[wid][self.rng.integers(len(self.pos[wid]))]
            y0 = int(np.clip(y - self.rng.integers(c), 0, H - c)); x0 = int(np.clip(x - self.rng.integers(c), 0, W - c))
        else:
            y0 = int(self.rng.integers(H - c + 1)); x0 = int(self.rng.integers(W - c + 1))
        a = img[:, y0:y0 + c, x0:x0 + c]; b = lab[y0:y0 + c, x0:x0 + c]
        k = int(self.rng.integers(4)); a = np.rot90(a, k, (1, 2)); b = np.rot90(b, k)
        if self.rng.random() < 0.5:
            a = a[:, :, ::-1]; b = b[:, ::-1]
        return np.ascontiguousarray(a), np.ascontiguousarray(b)


def normalise(img, mean, std):
    return (img - mean[:, None, None]) / std[:, None, None]


def loss_fn(logits, target, valid, class_w, dice_fraction):
    """Weighted CE on valid pixels + dice_fraction * (1 - soft Dice of the burned class)."""
    ce = F.cross_entropy(logits, target.clamp_max(N_CLASSES - 1), weight=class_w, reduction='none')
    ce = (ce * valid).sum() / valid.sum().clamp_min(1)
    p = logits.softmax(1)[:, 1]; t = (target == 1).float() * valid
    inter = (p * t * valid).sum(); denom = ((p * valid).sum() + t.sum())
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return (1 - dice_fraction) * ce + dice_fraction * dice


def evaluate_windows(model, windows, mean, std, device, tile, stride, batch):
    model.eval(); rows = []; tp = fp = fn = 0
    def predict(tiles):
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            x = torch.from_numpy(normalise(tiles, mean, std)).to(device)
            return model(x).float().cpu().numpy()
    for wid, (img, lab) in windows.items():
        valid = ~np.isnan(img).any(0) & (lab != IGNORE)
        lg = sliding_logits(predict, img, tile, stride, N_CLASSES, batch)
        pred = lg.argmax(0).astype(np.uint8)
        a, b, c, d = confusion(pred, lab, valid); P, R, Fv, I = prf(a, b, c)
        rows.append(dict(window_id=wid, tp=a, fp=b, fn=c, tn=d, P=P, R=R, F1=Fv, IoU=I)); tp += a; fp += b; fn += c
    f1s = [r['F1'] for r in rows if not math.isnan(r['F1'])]
    model.train()
    return rows, float(np.mean(f1s)) if f1s else float('nan'), prf(tp, fp, fn)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--manifest', default=None)
    ap.add_argument('--fold', required=True, choices=['F-AB', 'F-BC', 'F-SK']); ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='runs'); ap.add_argument('--epochs', type=int, default=120); ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--crops-per-epoch', type=int, default=2000); ap.add_argument('--crop', type=int, default=256); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--p-pos', type=float, default=0.5)
    ap.add_argument('--dice-fraction', type=float, default=0.5); ap.add_argument('--no-class-balancing', action='store_true')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--no-pretrained', action='store_true'); ap.add_argument('--sam-checkpoint', default=None)
    ap.add_argument('--tiny', action='store_true', help='smoke tests: tiny Mask2Former / SAM at 256 px'); ap.add_argument('--stride', type=int, default=128, help='sliding-inference stride recorded for eval.py (final scoring)')
    ap.add_argument('--val-stride', type=int, default=None, help='stride for the per-epoch validation pass (default = crop, i.e. no overlap, to keep epochs short; eval.py uses --stride)')
    ap.add_argument('--validation-json', default=None, help='JSON {fold: [window ids]} overriding the default reserved windows')
    ap.add_argument('--val-extra-per-province', type=int, default=2, help='extra validation windows per development province drawn (seeded) from the extra NBAC fires added by build_dataset.py --per-province; 0 = only the pre-registered reserved windows')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    set_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = read_manifest(a.manifest or os.path.join(a.data, 'manifest.csv'))
    rows = [r for r in rows if r.get('status', 'OK') == 'OK']
    val_map = json.load(open(a.validation_json)) if a.validation_json else DEFAULT_VALIDATION
    roles = fold_roles(rows, a.fold, val_map, a.val_extra_per_province, a.seed)
    train_ids = [w for w, r in roles.items() if r == 'TRAIN']; val_ids = [w for w, r in roles.items() if r == 'VALIDATION']
    assert train_ids and val_ids, roles
    assert not any(roles[w] == 'TEST' for w in train_ids + val_ids)
    run = f"{a.model}_{a.fold}{('_' + a.tag) if a.tag else ''}"; out = os.path.join(a.out, run); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    train_w = {w: read_window(a.data, w)[:2] for w in train_ids}
    val_w = {w: read_window(a.data, w)[:2] for w in val_ids}
    # normalisation and class weights from the TRAINING windows only
    samp = np.concatenate([img.reshape(9, -1)[:, ::97] for img, _ in train_w.values()], 1)
    mean = np.nanmean(samp, 1).astype(np.float32); std = (np.nanstd(samp, 1) + 1e-6).astype(np.float32)
    counts = np.array([sum(int(((lab == c) & ~np.isnan(img).any(0)).sum()) for img, lab in train_w.values()) for c in range(N_CLASSES)], np.float64)
    class_w = np.ones(N_CLASSES) if a.no_class_balancing else counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    class_w = class_w / class_w.mean()
    json.dump(dict(mean=mean.tolist(), std=std.tolist(), class_counts=counts.tolist(), class_weights=class_w.tolist(), bands=9), open(os.path.join(out, 'norm_stats.json'), 'w'), indent=1)
    cfg = dict(vars(a)); cfg.update(run=run, device=str(device), train_windows=train_ids, validation_windows=val_ids,
                                  test_windows=[w for w, r in roles.items() if r == 'TEST'], torch=torch.__version__, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    print(f'{run}: train {len(train_ids)} windows, validation {val_ids}, test (never read) {len(cfg["test_windows"])} | class weights {class_w.round(3)} | load {time.time() - t0:.0f}s', flush=True)
    model = build_model(a.model, 9, pretrained=not a.no_pretrained, sam_checkpoint=a.sam_checkpoint, tiny=a.tiny).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = max(1, a.crops_per_epoch // a.batch); total = a.epochs * steps_per_epoch; warm = 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))) * (1 - 1e-6 / a.lr) + 1e-6 / a.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    cw = torch.tensor(class_w, dtype=torch.float32, device=device)
    sampler = CropSampler(train_w, a.crop, a.p_pos, np.random.default_rng(a.seed))
    best = -1.0; bad = 0; logf = open(os.path.join(out, 'log.csv'), 'w', newline=''); lw = csv.writer(logf)
    lw.writerow(['epoch', 'train_loss', 'val_F1_mean_window', 'val_F1_pooled', 'lr', 'seconds', 'best_so_far']); logf.flush()
    for ep in range(1, a.epochs + 1):
        te = time.time(); model.train(); losses = []
        for _ in range(steps_per_epoch):
            xs, ys = zip(*[sampler.draw() for _ in range(a.batch)])
            x = torch.from_numpy(normalise(np.nan_to_num(np.stack(xs), nan=0.0), mean, std)).to(device)
            y = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
            valid = (~torch.isnan(torch.from_numpy(np.stack(xs))).any(1).to(device)) & (y != IGNORE)
            yv = torch.where(valid, y, torch.zeros_like(y))
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                if a.model == 'mask2former':
                    ml, cl = mask2former_targets(yv)
                    logits, loss = model(x, ml, cl)
                else:
                    logits = model(x); loss = loss_fn(logits.float(), yv, valid.float(), cw, a.dice_fraction)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss))
        rows_v, f1_mean, f1_pool = evaluate_windows(model, val_w, mean, std, device, a.crop, a.val_stride or a.crop, a.batch)
        improved = f1_mean > best + 0.001
        if improved:
            best = f1_mean; bad = 0; torch.save(dict(model=model.state_dict(), epoch=ep, val=rows_v), os.path.join(out, 'best.pt'))
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), epoch=ep), os.path.join(out, 'last.pt'))
        lw.writerow([ep, round(float(np.mean(losses)), 5), round(f1_mean, 4), round(f1_pool, 4), opt.param_groups[0]['lr'], round(time.time() - te, 1), round(best, 4)]); logf.flush()
        print(f'ep {ep:3d} loss {np.mean(losses):.4f} val F1 mean {f1_mean:.4f} pooled {f1_pool:.4f} ({time.time() - te:.0f}s){" *" if improved else ""}', flush=True)
        if bad >= a.patience:
            print(f'early stop at epoch {ep} (no improvement for {a.patience} epochs)'); break
    logf.close()
    ledger = dict(run=run, best_val_F1_mean_window=best, epochs_run=ep, best_pt_sha256=sha256(os.path.join(out, 'best.pt')), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                  wall_seconds=round(time.time() - t0), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    json.dump(ledger, open(os.path.join(out, 'ledger.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'run_ledger.csv'), 'a', newline='') as f:
        csv.writer(f).writerow([ledger[k] for k in ('run', 'best_val_F1_mean_window', 'epochs_run', 'best_pt_sha256', 'finished', 'wall_seconds', 'gpu')])
    print('done', ledger)


if __name__ == '__main__':
    main()
