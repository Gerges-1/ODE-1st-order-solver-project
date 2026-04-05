import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# عنوان التطبيق
st.title("Hospital Infection Spread Simulator with SIR Model")

# -----------------------------
# Sidebar: إدخال البيانات
st.sidebar.header("Input Parameters")

# Base Infection Rate
beta0 = st.sidebar.slider("Base Infection Rate (β₀)", 0.1, 2.0, 0.8)

# Environmental Factors
T = st.sidebar.slider("Temperature (°C)", 10, 40, 25)
H = st.sidebar.slider("Humidity (%)", 10, 100, 50)
V = st.sidebar.slider("Ventilation Speed (m/s)", 0.0, 5.0, 0.5)

# Indoor Social Factors
d = st.sidebar.slider("Distance between people (meters)", 0.5, 3.0, 1.0)

mask_eff = st.sidebar.slider("Mask Efficiency (%)", 0, 100, 60) / 100
sanitize_eff = st.sidebar.slider("Sanitization Efficiency (%)", 0, 100, 40) / 100

gamma = st.sidebar.slider("Recovery Rate (γ)", 0.1, 1.0, 0.3)

# Population
N = st.sidebar.number_input("Total People in Hospital", 10, 500, 100)
I0 = st.sidebar.number_input("Initial Infected People", 1, 20, 1)
R0_init = 0
S0 = N - I0 - R0_init

# -----------------------------
# Environmental Functions
def f_T(T):
    return 1 - 0.01 * (T - 25)

def f_H(H):
    return 1 - 0.005 * H

def f_V(V):
    return 1 + 0.1 * V

def f_d(d):
    k_d = 0.8
    return 1 / (1 + k_d * d)

# -----------------------------
# Effective Infection Rate
beta_eff = beta0 * f_T(T) * f_H(H) * f_V(V) * f_d(d) * (1 - mask_eff) * (1 - sanitize_eff)

# -----------------------------
# Effective R0
R0_eff = beta_eff / gamma

st.subheader("Calculated Values")
st.write(f"Effective Infection Rate (β): {beta_eff:.3f}")
st.write(f"Effective Basic Reproduction Number (R₀): {R0_eff:.3f}")

if R0_eff < 1:
    st.success("Low Risk: Disease will likely die out")
elif 1 <= R0_eff < 2:
    st.warning("Moderate Risk: Controlled spread")
else:
    st.error("High Risk: Rapid spread")

# -----------------------------
# SIR Model Differential Equations
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Time grid (days)
t_max = st.sidebar.slider("Simulation Days", 10, 200, 60)
t = np.linspace(0, t_max, t_max)

# Initial conditions vector
y0 = S0, I0, R0_init

# Integrate SIR equations
ret = odeint(deriv, y0, t, args=(N, beta_eff, gamma))
S, I, R = ret.T

# -----------------------------
# Plot results
st.subheader("SIR Model Simulation")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(t, S, 'b', alpha=0.7, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.7, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.7, lw=2, label='Recovered')
ax.set_xlabel('Days')
ax.set_ylabel('Number of People')
ax.set_title('Spread of Infection in Hospital')
ax.legend()
st.pyplot(fig)

# -----------------------------
# Key Observations
st.markdown("### Observations:")
st.write(f"- Peak Infected People: {int(max(I))} on day {int(t[np.argmax(I)])}")
st.write(f"- Total Infected by end: {int(R[-1]) + int(I[-1])}")
st.write("- Increasing distance, mask efficiency, sanitization, and ventilation reduces infection spread")
