import streamlit as st
import sympy as sp
from google import genai

# Setup the AI Model
try:
    ai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    ai_client = None


# Web Page Setup
st.set_page_config(page_title="ODE Classifier & Solver", layout="centered")

# Sidebar Sheet
with st.sidebar:
    st.header("Input Help Sheet")
    st.markdown(r"""
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
st.title("ODE Classifier & Solver")
st.markdown("Enter your equation parts based on the standard form:")
st.latex(r"M(x,y)dx + N(x,y)dy = 0")

# User Inputs & Live Preview
col1, col2, col3 = st.columns([1, 1, 1.5])
with col1:
    M_str = st.text_input("Enter M(x,y):", "(x + y)**2")
with col2:
    N_str = st.text_input("Enter N(x,y):", "-1")
with col3:
    st.markdown("**Live Preview:**")
    #try/except so the app doesn't crash if they are halfway through typing a parenthesis
    try:
        x_prev, y_prev = sp.symbols('x y', real=True)
        M_prev = sp.sympify(M_str, locals={'x': x_prev, 'y': y_prev})
        N_prev = sp.sympify(N_str, locals={'x': x_prev, 'y': y_prev})
        F_prev = sp.simplify(-M_prev / N_prev)
        
        # Displays dy/dx = ...
        st.latex(r"\frac{dy}{dx} = " + sp.latex(F_prev))
    except Exception:

        st.caption("Waiting for valid math...")

# The Solve
colA, colB = st.columns(2)
with colA:
    solve_clicked = st.button("Classify and Solve", type="primary")
with colB:
    use_ai = st.checkbox("Generate AI Steps")

if solve_clicked:
    x, y = sp.symbols('x y', real=True)
    
    try:
        # Parse inputs
        M = sp.sympify(M_str, locals={'x': x, 'y': y})
        N = sp.sympify(N_str, locals={'x': x, 'y': y})
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
        st.subheader("Identified Methods:")
        if class_methods:
            for name in sorted(class_methods):
                st.markdown(f"- **{name}**")
        else:
            st.warning("No method matched this equation.")

       
        # 4. Solve the ODE and render
        st.subheader("Final Solution:")
        with st.spinner("Calculating the integral..."):
            try:
                solution = sp.dsolve(ode, y_func)
                solution = sp.simplify(solution)
                
                if isinstance(solution, list):
                    for sol in solution:
                        st.latex(sp.latex(sol))
                else:
                    st.latex(sp.latex(solution))

                #AI STEPS FEATURE
                if use_ai and ai_client is not None:
                    st.divider()
                    st.subheader("Step-by-Step Breakdown")
                    with st.spinner("The AI is reverse-engineering the steps..."):
                        try:
                            prompt = f"""
                            Solve this differential equation: M(x,y)dx + N(x,y)dy = 0.
                            M = {M_str}
                            N = {N_str}
                            
                            CAS verified answer: {solution}
                            
                            CRITICAL INSTRUCTIONS FOR OUTPUT: 
                            1. DO NOT act like a teacher. Act like a strict, automated math calculator.
                            2. ZERO intro text, ZERO outro text, and ZERO conversational filler.
                            3. You must format your output STRICTLY using this exact spacing template for every step:
                               
                               **Step [Number]: [Action in 5 words or less]**
                               <br>
                               $$ [LaTeX math equation 1] $$
                               <br>
                               $$ [LaTeX math equation 2 (if needed)] $$
                               
                            4. THE HTML SPACING RULE: You MUST put the literal text "<br>" on its own line between the English text and the math. If a step has multiple equations, you MUST separate them with a comma and a "<br>" tag.
                            5. Your final answer must be the clean, human-readable real-number format. 
                            6. ABSOLUTELY DO NOT use or display complex exponentials, imaginary numbers, or Euler's formula anywhere in your text. 
                            
                            Provide ONLY the exact steps.
                            """
                            
                            # Using the new Google GenAI syntax
                            response = ai_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt
                            )
                            st.markdown(response.text)
                        except Exception as ai_error:
                            st.error(f"🔍 AI ERROR DETECTED: {ai_error}")
                # ----------------------------
            except NotImplementedError:
                st.error("SymPy could not compute the final integral for this equation.")
            except Exception as math_error:
                st.error(f"🔍 MATH ERROR DETECTED: {math_error}")

    except sp.SympifyError:
        st.error("⚠️ Error: Could not understand the math. Please check your spelling.")

st.divider()
st.caption("Built with Python, SymPy, and Streamlit.")
