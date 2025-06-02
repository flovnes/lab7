import numpy as np
import matplotlib.pyplot as plt

def df(t, y_val):
    return 2 * t * y_val**2

def y(t_val):
    return 1 / (1 - t_val**2)

def rk(df_func, t_curr, y_curr, h_step):
    f1 = df_func(t_curr, y_curr)
    f2 = df_func(t_curr + 0.5 * h_step, y_curr + 0.5 * h_step * f1)
    f3 = df_func(t_curr + 0.5 * h_step, y_curr + 0.5 * h_step * f2)
    f4 = df_func(t_curr + h_step, y_curr + h_step * f3)
    y_next_val = y_curr + h_step/6 * (f1 + 2*f2 + 2*f3 + f4) 
    return y_next_val

def solve(t0_solve, y0_solve, h_solve):
    tt = [t0_solve]
    yy = [y0_solve]
    
    t_iter = t0_solve 
    y_iter_val = y0_solve 

    for _ in range(n): 
        if t_iter + h_solve > max_t: 
            break
        
        y_next_rk = rk(df, t_iter, y_iter_val, h_solve) 

        t_iter += h_solve
        y_iter_val = y_next_rk

        tt.append(t_iter)
        yy.append(y_iter_val)
            
    return tt, yy

t0 = 0.0
y0 = 1.0
h_b = 0.1
n = 10
max_t = 0.99

print(f"y' = 2*t*y^2, y({t0:.0f}) = {y0:.0f}, h = {h_b}\n")

t_pts, y_pts = solve(t0, y0, h_b)

tn_raw = np.array(t_pts)
yn_raw = np.array(y_pts)
    
ya_comp = np.array([y(tv) for tv in tn_raw])
    
valid_analytical_indices = np.isfinite(ya_comp)
tn_final = tn_raw[valid_analytical_indices]
yn_final = yn_raw[valid_analytical_indices]
ya_final = ya_comp[valid_analytical_indices]

abs_err = np.abs(yn_final - ya_final)
    
max_residual = np.max(abs_err)
print(f"Макс. похибка : {max_residual:.4e}")


max_y_plot = 15.0

fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.suptitle(f"y' = 2*t*y^2, y({t0:.0f})={y0:.0f} (h={h_b})", fontsize=14)

max_t_p1 = t0
if t_pts:
    max_t_p1 = max(max_t_p1, t_pts[-1])

t_end_an_g1 = min(max_t_p1 + h_b, max_t + h_b * 0.1, 0.999)
if t_end_an_g1 <= t0:
    t_end_an_g1 = max_t if max_t > t0 else t0 + h_b

ta1 = np.linspace(t0, t_end_an_g1, 300)
ya1_raw = np.array([y(ti) for ti in ta1]) 
valid_an1 = np.where(np.isfinite(ya1_raw) & (np.abs(ya1_raw) < max_y_plot * 1.1))[0]

if len(valid_an1) > 0:
    ax1.plot(ta1[valid_an1], ya1_raw[valid_an1], 'k--', label='Точний розв\'язок', lw=1.5, zorder=1)

tp_raw_plot = np.array(t_pts)
yp_raw_plot = np.array(y_pts)
    
valid_plot_idx = np.where(np.isfinite(yp_raw_plot) & (np.abs(yp_raw_plot) < max_y_plot * 1.5))[0] 
tp_plot = tp_raw_plot[valid_plot_idx]
yp_plot = yp_raw_plot[valid_plot_idx]

if len(tp_plot) > 0:
    ax1.plot(tp_plot, yp_plot, "b-", marker="o", label="Метод Рунге-Кутти", ms=6, zorder=2) 
else: 
    ax1.plot([],[], "b-", marker="o", label="Метод Рунге-Кутти")

ax1.set_xlabel('t')
ax1.set_ylabel('y(t)')
ax1.legend(loc='upper left')
ax1.grid(True, ls=':', alpha=0.7)
ax1.set_ylim(bottom=min(y0 - 2, -1), top=max_y_plot) 
ax1.set_xlim(left=t0 - h_b*0.1, right=max(t_end_an_g1 + h_b*0.1, t0 + h_b))
fig1.tight_layout(rect=[0, 0, 1, 0.96]) 

plt.show()