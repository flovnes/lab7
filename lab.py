import numpy as np

def dy_dt(t, y):
    return 2*t*y**2

def analytical_solution(t):
    return 1/(1-t**2)

def rk4_step(f, t_i, y_i, h):
    k1 = h * f(t_i, y_i)
    k2 = h * f(t_i + 0.5 * h, y_i + 0.5 * k1)
    k3 = h * f(t_i + 0.5 * h, y_i + 0.5 * k2)
    k4 = h * f(t_i + h, y_i + k3)
    y_next = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return y_next

t0 = 0.0
y0 = 1.0
h = 0.1

N_abm_steps = 5

t_values = [t0]
y_values = [y0]
f_values = [dy_dt(t0, y0)]

print(f"y' = y - t^2 + 1, y({t0}) = {y0}, h = {h}\n")
print(f"t = {t_values[0]:.2f}, y = {y_values[0]:.6f}, y_a = {analytical_solution(t_values[0]):.6f}, f(t,y) = {f_values[0]:.6f}")

current_t = t0
current_y = y0

for i in range(3):
    y_next = rk4_step(dy_dt, current_t, current_y, h)
    current_t += h
    current_y = y_next

    t_values.append(round(current_t, 10))
    y_values.append(current_y)
    f_values.append(dy_dt(current_t, current_y))

    print(f"крок {i+1}: t = {t_values[-1]:.2f}, y = {y_values[-1]:.6f}, y_a = {analytical_solution(t_values[-1]):.6f}, f(t,y) = {f_values[-1]:.6f}")


for i in range(N_abm_steps):
    k = len(y_values) - 1

    # y*_{i+1} = y_i + (h/24) * (55*f_i - 59*f_{i-1} + 37*f_{i-2} - 9*f_{i-3})
    y_predicted = y_values[k] + (h / 24.0) * (
        55 * f_values[k] -
        59 * f_values[k-1] +
        37 * f_values[k-2] -
        9 * f_values[k-3]
    )
    
    t_next = round(t_values[k] + h, 10)
    f_predicted = dy_dt(t_next, y_predicted)

    # y_{i+1} = y_i + (h/24) * (9*f*_{i+1} + 19*f_i - 5*f_{i-1} + f_{i-2})
    y_corrected = y_values[k] + (h / 24.0) * (
        9 * f_predicted +
        19 * f_values[k] -
        5 * f_values[k-1] +
        1 * f_values[k-2]
    )
    
    current_t = t_next
    current_y = y_corrected

    t_values.append(current_t)
    y_values.append(current_y)
    f_values.append(dy_dt(current_t, current_y))

    print(f"  крок {i+1}: t = {t_values[-1]:.2f}, y_pred = {y_predicted:.6f}, y_corr = {y_values[-1]:.6f}, y_analyt = {analytical_solution(t_values[-1]):.6f}, f(t,y) = {f_values[-1]:.6f}")

print("-" * 60)
print("Всі обчислені точки:")
print("  t  |  y             |  y (точне значення) | похибка | f(t,y)")
for i in range(len(t_values)):
    t = t_values[i]
    y_num = y_values[i]
    y_an = analytical_solution(t)
    error = abs(y_num - y_an)
    f_val = f_values[i]
    print(f"{t:4.2f} | {y_num:14.6f} | {y_an:19.6f} | {error:8.2e} | {f_val:8.6f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, 'bo-', label='Розв\'язок')

t_analytical_fine = np.linspace(t_values[0], t_values[-1], 200)
y_analytical_fine = [analytical_solution(ti) for ti in t_analytical_fine]
plt.plot(t_analytical_fine, y_analytical_fine, 'r--', label='Графік функції')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f"Розв'язок задачі Коші для y' = 2*t*y^2, y({t0:.0f})={y0:.0f}, h={h}")
plt.legend()
plt.grid(True)
plt.show()