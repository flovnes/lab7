import numpy as np
import matplotlib.pyplot as plt

def df(t, y):
    return 2 * t * y**2

def y(t):
    return 1 / (1 - t**2)

def rk(df, t, y, h):
    f1 = df(t, y)
    f2 = df(t + 0.5 * h, y + 0.5 * h * f1)
    f3 = df(t + 0.5 * h, y + 0.5 * h * f2)
    f4 = df(t + h, y + h * f3)
    y_next = y + h/6 * (f1 + 2*f2 + 2*f3 + f4) 
    return y_next

def solve(t0, y0, h, variant):
    tt = [t0]
    yy = [y0]
    ff = []

    f_at_t0_y0 = df(t0, y0)
    ff.append(f_at_t0_y0)

    t = t0
    y = y0
    n_start = 3 # 4-1

    for _ in range(n_start):
        if t + h > max_t: break
        
        y_next_rk = rk(df, t, y, h)

        t += h
        y = y_next_rk

        tt.append(t)
        yy.append(y)
        ff.append(df(t, y))

    p_prev = None
    y_prev = None

    # n кроків методом абрамса
    
    for _ in range(n):
        k = len(yy) - 1 
        if tt[k] + h > max_t: break
        
        p = yy[k] + h/24 * (-9*ff[k-3] + 37*ff[k-2] - 59*ff[k-1] + 55*ff[k])
        
        t_next = tt[k] + h
        
        if variant:
            # якшо не перший крок
            if p_prev is not None:
                p -= (19/270) * (p_prev - y_prev)
        
        y_next = yy[k] + h/24 * (ff[k-2] - 5*ff[k-1] + 19*ff[k] + 9*df(t_next, p))
        
        p_prev = p
        y_prev = y_next

        t = t_next
        y = y_next

        tt.append(t)
        yy.append(y)
        ff.append(df(t, y))
            
    return tt, yy, ff

t0 = 0.0
y0 = 1.0
h_b = 0.01
n = 100
max_t = 0.99

print(f"y' = 2*t*y^2, y({t0:.0f}) = {y0:.0f}, h = {h_b}\n")

sc_params = [
    {"id": "a", "lbl": "а) Без управляючого параметра, h'=h", "h_v": h_b, "use_cp": False, "ls": "g-.", "mkr": "^"},
    {"id": "b", "lbl": "б) З управляючим параметром, h'=h",   "h_v": h_b, "use_cp": True,  "ls": "b-",  "mkr": "o"},
    {"id": "c", "lbl": "в) Без управляючого параметра, h'=2*h","h_v": 2*h_b, "use_cp": False, "ls": "m--", "mkr": "s"},
    {"id": "d", "lbl": "г) З управляючим параметром, h'=2*h",  "h_v": 2*h_b, "use_cp": True,  "ls": "c:",  "mkr": "x"},
]

results = {}

for p in sc_params:
    t_pts, y_pts, f_vs = solve(t0, y0, h=p["h_v"], variant=p["use_cp"])
    results[p["id"]] = {
        "lbl": p["lbl"], "t": t_pts, "y": y_pts, "f": f_vs,
        "ls": p["ls"], "mkr": p["mkr"], "h_v": p["h_v"],
        "use_cp_v": p["use_cp"]
    }

print("Порівняння точності:")
for cfg in sc_params:
    s_id = cfg["id"]
    res = results[s_id]

    tn_raw = np.array(res['t'])
    yn_raw = np.array(res['y'])
    
    tn = tn_raw
    yn = yn_raw
    
    ya_comp = np.array([y(tv) for tv in tn])
    
    valid_analytical_indices = np.isfinite(ya_comp)
    tn_final = tn[valid_analytical_indices]
    yn_final = yn[valid_analytical_indices]
    ya_final = ya_comp[valid_analytical_indices]

    abs_err = np.abs(yn_final - ya_final)
    
    max_err = np.max(abs_err)
    print(f"{res['lbl']}: {max_err:.4e}")

max_y_plot = 15.0

fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.suptitle(f"y' = 2*t*y^2, y({t0:.0f})={y0:.0f} (h={h_b})", fontsize=14)

max_t_p1 = t0
rel_ids_g1 = ["a", "b"]
for s_id in rel_ids_g1:
    res = results[s_id]
    if res['t']: max_t_p1 = max(max_t_p1, res['t'][-1])

t_end_an_g1 = min(max_t_p1 + h_b, max_t + h_b * 0.1, 0.999)
if t_end_an_g1 <= t0: t_end_an_g1 = max_t if max_t > t0 else t0 + h_b

ta1 = np.linspace(t0, t_end_an_g1, 300)
ya1_raw = np.array([y(ti) for ti in ta1])
valid_an1 = np.where(np.isfinite(ya1_raw) & (np.abs(ya1_raw) < max_y_plot * 1.1))[0]
if len(valid_an1) > 0:
    ax1.plot(ta1[valid_an1], ya1_raw[valid_an1], 'k--', label='Точний розв\'язок', lw=1.5, zorder=1)

for s_id in rel_ids_g1:
    r = results[s_id]
    tp_raw = np.array(r['t'])
    yp_raw = np.array(r['y'])
    
    valid_plot_idx = np.where(np.isfinite(yp_raw) & (np.abs(yp_raw) < max_y_plot * 1.5))[0] 
    tp = tp_raw[valid_plot_idx]
    yp = yp_raw[valid_plot_idx]

    if len(tp) > 0:
        z = 2 + int(r["use_cp_v"])
        ax1.plot(tp, yp, r['ls'], marker=r['mkr'], label=r['lbl'], ms=6, zorder=z) 
    else: 
        ax1.plot([],[], r['ls'], marker=r['mkr'], label=r['lbl'])

ax1.set_xlabel('t')
ax1.set_ylabel('y(t)')
ax1.legend(loc='upper left')
ax1.grid(True, ls=':', alpha=0.7)
ax1.set_ylim(bottom=min(y0 - 2, -1), top=max_y_plot) 
ax1.set_xlim(left=t0 - h_b*0.1, right=max(t_end_an_g1 + h_b*0.1, t0 + h_b))
fig1.tight_layout(rect=[0, 0, 1, 0.96]) 

fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.suptitle(f"y' = 2*t*y^2, y({t0:.0f})={y0:.0f} (h={h_b})", fontsize=14)

max_t_p2 = t0
rel_ids_g2 = ["c", "d"]
h_p2 = 2 * h_b 
for s_id in rel_ids_g2:
    res = results[s_id]
    if res['t']: max_t_p2 = max(max_t_p2, res['t'][-1])

t_end_an_g2 = min(max_t_p2 + h_p2, max_t + h_p2 * 0.1, 0.999)
if t_end_an_g2 <= t0: t_end_an_g2 = max_t if max_t > t0 else t0 + h_p2

ta2 = np.linspace(t0, t_end_an_g2, 300)
ya2_raw = np.array([y(ti) for ti in ta2])
valid_an2 = np.where(np.isfinite(ya2_raw) & (np.abs(ya2_raw) < max_y_plot * 1.1))[0]
if len(valid_an2) > 0:
    ax2.plot(ta2[valid_an2], ya2_raw[valid_an2], 'k--', label='Точний розв\'язок', lw=1.5, zorder=1)

for s_id in rel_ids_g2:
    r = results[s_id]
    tp_raw = np.array(r['t'])
    yp_raw = np.array(r['y'])
    valid_plot_idx = np.where(np.isfinite(yp_raw) & (np.abs(yp_raw) < max_y_plot * 1.5))[0] 
    tp = tp_raw[valid_plot_idx]
    yp = yp_raw[valid_plot_idx]
    if len(tp) > 0:
        z = 2 + int(r["use_cp_v"])
        ax2.plot(tp, yp, r['ls'], marker=r['mkr'], label=r['lbl'], ms=6, zorder=z)
    else:
        ax2.plot([],[], r['ls'], marker=r['mkr'], label=r['lbl'])

ax2.set_xlabel('t')
ax2.set_ylabel('y(t)')
ax2.legend(loc='upper left')
ax2.grid(True, ls=':', alpha=0.7)
ax2.set_ylim(bottom=min(y0 - 2, -1), top=max_y_plot)
ax2.set_xlim(left=t0 - h_p2*0.1, right=max(t_end_an_g2 + h_p2*0.1, t0 + h_p2))
fig2.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()