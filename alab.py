import numpy as np
import matplotlib.pyplot as plt

def dy_dt(t, y):
    val = 2 * t * y**2
    return val

def analytical_solution(t):
    return 1.0 / (1.0 - t**2)

def rk_step(f, t_i, y_i, h_step):
    k1 = h_step * f(t_i, y_i)
    k2 = h_step * f(t_i + 0.5 * h_step, y_i + 0.5 * k1)
    k3 = h_step * f(t_i + 0.5 * h_step, y_i + 0.5 * k2)
    k4 = h_step * f(t_i + h_step, y_i + k3)
    y_next = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return y_next

def solve(t0_sc, y0_sc, h_sc, n_abm_sc, use_corrector_sc,
                   dy_dt_func, rk_step_func, max_t_limit):
    t_points = [t0_sc]
    y_points = [y0_sc]
    f_points = []

    f0_val = dy_dt_func(t0_sc, y0_sc)
    if not np.isfinite(f0_val):
        f_points.append(np.nan)
        return t_points, y_points, f_points
    f_points.append(f0_val)

    current_t_sc = t0_sc
    current_y_sc = y0_sc
    num_rk_startup_steps = 3

    # метoд Рунге-Кутта
    for i in range(num_rk_startup_steps):
        if current_t_sc + h_sc > max_t_limit + 1e-9:
            break
        if not np.isfinite(current_y_sc):
            break

        y_next_sc = rk_step_func(dy_dt_func, current_t_sc, current_y_sc, h_sc)

        current_t_sc = round(current_t_sc + h_sc, 10)
        current_y_sc = y_next_sc

        t_points.append(current_t_sc)
        y_points.append(current_y_sc)

        if np.isfinite(current_y_sc):
            f_curr_val = dy_dt_func(current_t_sc, current_y_sc)
            f_points.append(f_curr_val if np.isfinite(f_curr_val) else np.nan)
        else:
            f_points.append(np.nan)
            break

    # метод Адамса
    for _ in range(n_abm_sc):
        k = len(y_points) - 1 

        if t_points[k] + h_sc > max_t_limit + 1e-9: 
            break
        if not np.isfinite(y_points[k]) or \
           not all(np.isfinite(f_points[j]) for j in range(k-3, k+1)): 
            break

        y_predicted_sc = y_points[k] + (h_sc / 24.0) * (
            55 * f_points[k] - 59 * f_points[k-1] + 37 * f_points[k-2] - 9 * f_points[k-3]
        )

        t_next_sc = round(t_points[k] + h_sc, 10)
        current_y_abm_step = y_predicted_sc

        if use_corrector_sc:
            if not np.isfinite(y_predicted_sc):
                f_predicted_val = np.nan
            else:
                f_predicted_val = dy_dt_func(t_next_sc, y_predicted_sc)
            
            if not np.isfinite(f_predicted_val): 
                pass 
            else:
                y_corrected_sc = y_points[k] + (h_sc / 24.0) * (
                    9 * f_predicted_val + 19 * f_points[k] - 5 * f_points[k-1] + f_points[k-2]
                )
                current_y_abm_step = y_corrected_sc

        current_t_sc = t_next_sc
        current_y_sc = current_y_abm_step
        t_points.append(current_t_sc)
        y_points.append(current_y_sc)

        if np.isfinite(current_y_sc):
            f_curr_val = dy_dt_func(current_t_sc, current_y_sc)
            f_points.append(f_curr_val if np.isfinite(f_curr_val) else np.nan)
        else:
            f_points.append(np.nan)
            break

    return t_points, y_points, f_points

t_initial = 0.0
y_initial = 1.0
h_base_step = 0.1
n_abm_steps = 5
max_t_calculation = 0.99

print(f"Задача Коші: y' = 2*t*y^2, y({t_initial}) = {y_initial} h = {h_base_step}")

scenarios_params = [
    {"id": "a", "label": "а) Без коректора, h=h_base", "h_val": h_base_step, "use_corr": False, "linestyle": "g-.", "marker": "^"},
    {"id": "b", "label": "б) З коректором, h=h_base",   "h_val": h_base_step, "use_corr": True,  "linestyle": "b-",  "marker": "o"},
    {"id": "c", "label": "в) Без коректора, h=2*h_base","h_val": 2*h_base_step, "use_corr": False, "linestyle": "m--", "marker": "s"},
    {"id": "d", "label": "г) З коректором, h=2*h_base",  "h_val": 2*h_base_step, "use_corr": True,  "linestyle": "c:",  "marker": "x"},
]

all_results = {}

for params in scenarios_params:
    time_pts, y_pts, f_vals = solve(
        t0_sc=t_initial, y0_sc=y_initial,
        h_sc=params["h_val"],
        n_abm_sc=n_abm_steps,
        use_corrector_sc=params["use_corr"],
        dy_dt_func=dy_dt,
        rk_step_func=rk_step,
        max_t_limit=max_t_calculation
    )
    all_results[params["id"]] = {
        "label": params["label"], "t": time_pts, "y": y_pts, "f": f_vals,
        "linestyle": params["linestyle"], "marker": params["marker"], "h_val": params["h_val"]
    }


print("\nПорівняння точності\n")
for scenario_config in scenarios_params:
    sc_id = scenario_config["id"]
    res = all_results[sc_id]

    t_numerical = np.array(res['t'])
    y_numerical = np.array(res['y'])

    y_analytical_at_t_comparison = np.array([analytical_solution(t_val) for t_val in t_numerical])

    abs_errors = np.abs(y_numerical - y_analytical_at_t_comparison)
    max_abs_error = np.max(abs_errors)
    print(f"{res['label']}: {max_abs_error:.4e}")


max_y_display = 7.0

# Графік 1
fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.suptitle(f"Розв'язки: y' = 2*t*y^2, y({t_initial:.1f})={y_initial:.1f} (h_base={h_base_step})", fontsize=14)
ax1.set_title("Варіанти (а) та (б)", fontsize=12)

max_t_for_plot1 = t_initial
relevant_ids_g1 = ["a", "b"]
for sc_id in relevant_ids_g1:
    res = all_results[sc_id]
    if res['t'] and np.isfinite(res['t'][-1]): max_t_for_plot1 = max(max_t_for_plot1, res['t'][-1])

t_end_analytical_g1 = min(max_t_for_plot1 + h_base_step, max_t_calculation + h_base_step, 0.999)
if t_end_analytical_g1 <= t_initial : t_end_analytical_g1 = max_t_calculation if max_t_calculation > t_initial else t_initial + h_base_step

t_analytical1 = np.linspace(t_initial, t_end_analytical_g1, 300)
y_analytical1_raw = np.array([analytical_solution(ti) for ti in t_analytical1])

valid_an_idx1 = np.where(np.isfinite(y_analytical1_raw) & (np.abs(y_analytical1_raw) < max_y_display * 1.1))[0]
if len(valid_an_idx1) > 0:
    ax1.plot(t_analytical1[valid_an_idx1], y_analytical1_raw[valid_an_idx1],
                'k--', label=f'Точний розв\'язок', linewidth=1.5, zorder=1)

for sc_id in relevant_ids_g1:
    result = all_results[sc_id]
    t_plot = np.array(result['t'])
    y_plot = np.array(result['y'])
    valid_idx = np.where(np.isfinite(y_plot) & (np.abs(y_plot) < max_y_display * 1.5))[0]
    if len(valid_idx) > 0:
        ax1.plot(t_plot[valid_idx], y_plot[valid_idx],
                    result['linestyle'], marker=result['marker'],
                    label=result['label'], markersize=6, zorder=int(sc_id == 'b') + 2) # Corrector on top
    else:
            ax1.plot([],[], result['linestyle'], marker=result['marker'], label=result['label'])

ax1.set_xlabel('t')
ax1.set_ylabel('y(t)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(min(0, y_initial - 1 if y_initial > 0 else y_initial -2 ), max_y_display) # Adjusted ylim
ax1.set_xlim(t_initial - h_base_step*0.1, t_end_analytical_g1 + h_base_step*0.1)
fig1.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusted rect for suptitle

# Графік 2
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.suptitle(f"Розв'язки: y' = 2*t*y^2, y({t_initial:.1f})={y_initial:.1f} (h_base={h_base_step})", fontsize=14)
ax2.set_title("Варіанти (в) та (г)", fontsize=12)

max_t_for_plot2 = t_initial
relevant_ids_g2 = ["c", "d"]
h_for_xlim_g2 = 2 * h_base_step
for sc_id in relevant_ids_g2:
    res = all_results[sc_id]
    if res['t'] and np.isfinite(res['t'][-1]): max_t_for_plot2 = max(max_t_for_plot2, res['t'][-1])

t_end_analytical_g2 = min(max_t_for_plot2 + h_for_xlim_g2, max_t_calculation + h_for_xlim_g2, 0.999)
if t_end_analytical_g2 <= t_initial : t_end_analytical_g2 = max_t_calculation if max_t_calculation > t_initial else t_initial + h_for_xlim_g2

t_analytical2 = np.linspace(t_initial, t_end_analytical_g2, 300)
y_analytical2_raw = np.array([analytical_solution(ti) for ti in t_analytical2])
valid_an_idx2 = np.where(np.isfinite(y_analytical2_raw) & (np.abs(y_analytical2_raw) < max_y_display * 1.1))[0]
if len(valid_an_idx2) > 0:
    ax2.plot(t_analytical2[valid_an_idx2], y_analytical2_raw[valid_an_idx2],
                'k--', label=f'Точний розв\'язок', linewidth=1.5, zorder=1)

for sc_id in relevant_ids_g2:
    result = all_results[sc_id]
    t_plot = np.array(result['t'])
    y_plot = np.array(result['y'])
    valid_idx = np.where(np.isfinite(y_plot) & (np.abs(y_plot) < max_y_display * 1.5))[0]
    if len(valid_idx) > 0:
        ax2.plot(t_plot[valid_idx], y_plot[valid_idx],
                    result['linestyle'], marker=result['marker'],
                    label=result['label'], markersize=6, zorder=int(sc_id == 'd') + 2) # Corrector on top
    else:
            ax2.plot([],[], result['linestyle'], marker=result['marker'], label=result['label'])
            ax2.set_xlabel('t')
ax2.set_ylabel('y(t)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.set_ylim(min(0, y_initial -1 if y_initial > 0 else y_initial -2 ), max_y_display) # Adjusted ylim
ax2.set_xlim(t_initial - h_for_xlim_g2*0.1, t_end_analytical_g2 + h_for_xlim_g2*0.1)
fig2.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusted rect for suptitle

plt.show()