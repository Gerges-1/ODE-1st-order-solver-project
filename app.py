import streamlit as st
import sympy as sp

# Web Page Setup
st.set_page_config(page_title="ODE Classifier & Solver", page_icon="🧮", layout="centered")

# Sidebar Sheet
with st.sidebar:
    st.header("Input Help Sheet")
    st.markdown("""
    Use these formats when typing your equations into the text boxes:

    **Basic Math:**
    * Multiplication: `x * y`
    * Power ($x^n$): `x**n`
    * Euler's $e^x$: `exp(x)` or `E**x`

    **Logarithms:**
    * Natural Log ($\ln x$): `log(x)`
    * Base-10 Log ($\log_{10} x$): `log(x, 10)`
    * Any Base ($\log_b x$): `log(x, b)`
    
    **Roots:**
    * Square Root ($\sqrt{x}$): `sqrt(x)`
    * $n$-th Root ($\sqrt[n]{x}$): `x**(1/n)` 
    
    **Trigonometry:**
    * Regular: `sin(x)`, `cos(x)`, `tan(x)`
    * Inverse ($\sin^{-1}x$): `asin(x)`, `acos(x)`, `atan(x)`
    
    **Hyperbolic:**
    * Regular: `sinh(x)`, `cosh(x)`, `tanh(x)`
    * Inverse: `asinh(x)`, `acosh(x)`, `atanh(x)`
    
    **⚠️ Important Rule:**
    Always use parentheses for functions! 
    * **Right:** `sin(x) * cos(y)`
    * **Wrong:** `sin x * cos y`
    """)

# Main
st.title("🧮 ODE Classifier & Solver")
st.markdown("Enter your equation parts based on the standard form:")
st.latex(r"M(x,y)dx + N(x,y)dy = 0")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    M_str = st.text_input("Enter M(x,y):", "cos(x + y + 1)**2 + 1")
with col2:
    N_str = st.text_input("Enter N(x,y):", "-1")

# The Solve
if st.button("Classify and Solve", type="primary"):
    x, y = sp.symbols('x y')
    
    try:
        # Parse inputs
        M = sp.sympify(M_str)
        N = sp.sympify(N_str)
        F = sp.simplify(-M / N)
        y_func = sp.Function('y')(x)
        ode = sp.Eq(y_func.diff(x), F.subs(y, y_func))
        
        # 1. Classify the ODE (Standard SymPy checks)
        sympy_methods = sp.classify_ode(ode)
        class_methods = set()

        for method in sympy_methods:
            method_lower = method.lower()
            if 'separable' in method_lower:
                class_methods.add("Separation of Variables")
            elif 'linear' in method_lower:
                class_methods.add("Linear First Order")
            elif 'homogeneous' in method_lower and 'rational' not in method_lower:
                class_methods.add("Homogeneous")
            elif 'bernoulli' in method_lower:
                class_methods.add("Non-linear 'Bernoulli Equation'")
            elif 'homogeneous_rational' in method_lower:
                class_methods.add("Non-homogeneous (Rational Fraction)")

        # 2.Math Checks for Exact / Non-Exact
        My = sp.diff(M, y)
        Nx = sp.diff(N, x)
        
        if sp.simplify(My - Nx) == 0:
            class_methods.add("Exact ODE")
        else:
            diff_MN = sp.simplify(My - Nx)
            if not sp.simplify(diff_MN / N).has(y):
                class_methods.add("Non-Exact (Integrating factor depends on x)")
            elif not sp.simplify(diff_MN / M).has(x):
                class_methods.add("Non-Exact (Integrating factor depends on y)")

        # 3.Math Check for Reduced to Separation
        Fx = sp.diff(F, x)
        Fy = sp.diff(F, y)
        if Fx != 0 and Fy != 0:
            ratio = sp.simplify(Fx / Fy)
            if ratio.is_constant() and ratio != 0:
                class_methods.add("Reduced to Separation: f(ax+by+c)")

        # Display the classification results
        st.subheader("📌 Identified Methods 📌:")
        if class_methods:
            for name in sorted(class_methods):
                st.markdown(f"- **{name}**")
        else:
            st.warning("No method matched this equation.")

        # 4. Solve the ODE and render
        st.subheader("✅ Final Solution ✅:")
        with st.spinner("Calculating the integral..."):
            try:
                solution = sp.dsolve(ode, y_func)
                
                if isinstance(solution, list):
                    for sol in solution:
                        st.latex(sp.latex(sol))
                else:
                    st.latex(sp.latex(solution))
            except NotImplementedError:
                st.error("SymPy could not compute the final integral for this equation.")
            except Exception as e:
                st.error("Could not solve due to a complex math error.")

    except sp.SympifyError:
        st.error("⚠️ Error: Could not understand the math. Please check your spelling.")

st.divider()
st.caption("Built with Python, SymPy, and Streamlit.")
