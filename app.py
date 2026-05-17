import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="RoboAdvisor Crediticio",
    layout="wide"
)

# =====================================================
# TÍTULO
# =====================================================

st.title("🏦 RoboAdvisor Crediticio Avanzado")

st.write("""
Sistema automatizado de evaluación financiera
utilizando Machine Learning y reglas bancarias.
  Grupo 3 - Claudio Andrés Varas Pinchulef - 
Pedro Esau Sierra Romero - Tomás Ignacio Bao Piraud.

""")

# =====================================================
# FUNCIÓN LIMPIAR NÚMEROS
# =====================================================

def limpiar_numero(valor):

    valor = valor.replace(".", "")
    valor = valor.replace(",", "")

    if valor == "":
        return 0

    return int(valor)

# =====================================================
# BASE DE DATOS SIMULADA
# =====================================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({

    "ingreso": np.random.randint(
        500000,
        5000000,
        n
    ),

    "deuda": np.random.randint(
        50000,
        3000000,
        n
    ),

    "moras": np.random.randint(
        0,
        6,
        n
    ),

    "antiguedad": np.random.randint(
        0,
        15,
        n
    )
})

# =====================================================
# VARIABLE OBJETIVO
# =====================================================

df["riesgo"] = (

    (df["deuda"] / df["ingreso"] > 0.5)

    |

    (df["moras"] > 3)

    |

    (df["antiguedad"] < 1)

).astype(int)

# =====================================================
# MACHINE LEARNING
# =====================================================

X = df[[
    "ingreso",
    "deuda",
    "moras",
    "antiguedad"
]]

y = df["riesgo"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo = LogisticRegression()

modelo.fit(X_train, y_train)

# =====================================================
# IMPORTANCIA VARIABLES
# =====================================================

coeficientes = pd.DataFrame({

    "Variable": X.columns,

    "Coeficiente": modelo.coef_[0]

})

# =====================================================
# EVALUACIÓN DEL MODELO
# =====================================================

predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predicciones
)

matriz = confusion_matrix(
    y_test,
    predicciones
)

# =====================================================
# FORMULARIO
# =====================================================

st.header("📋 Perfil Financiero del Cliente")

col1, col2 = st.columns(2)

# =====================================================
# COLUMNA 1
# =====================================================

with col1:

    edad = st.number_input(
        "Edad",
        min_value=18,
        max_value=100,
        value=30
    )

    ingreso_texto = st.text_input(
        "Ingreso mensual ($)",
        value="1.500.000"
    )

    ingreso = limpiar_numero(ingreso_texto)

    deuda_texto = st.text_input(
        "Deuda mensual ($)",
        value="300.000"
    )

    deuda = limpiar_numero(deuda_texto)

    ahorros_texto = st.text_input(
        "Ahorros disponibles ($)",
        value="5.000.000"
    )

    ahorros = limpiar_numero(ahorros_texto)

    moras = st.number_input(
        "Cantidad de moras",
        min_value=0,
        value=0
    )

# =====================================================
# COLUMNA 2
# =====================================================

with col2:

    antiguedad = st.number_input(
        "Años de antigüedad laboral",
        min_value=0,
        value=3
    )

    contrato = st.selectbox(
        "Tipo de contrato",
        [
            "Indefinido",
            "Plazo fijo",
            "Honorarios"
        ]
    )

    vivienda = st.selectbox(
        "Tipo de vivienda",
        [
            "Propia",
            "Arrendada",
            "Familiar"
        ]
    )

    estado_civil = st.selectbox(
        "Estado civil",
        [
            "Soltero",
            "Casado",
            "Divorciado"
        ]
    )

    dependientes = st.number_input(
        "Personas dependientes",
        min_value=0,
        value=0
    )

# =====================================================
# CRÉDITO SOLICITADO
# =====================================================

monto_texto = st.text_input(
    "Monto solicitado ($)",
    value="10.000.000"
)

monto_credito = limpiar_numero(monto_texto)

cuotas = st.number_input(
    "Cantidad de cuotas",
    min_value=1,
    max_value=84,
    value=36
)

# =====================================================
# BOTÓN
# =====================================================

if st.button("📊 Evaluar Cliente"):

    # =================================================
    # CLIENTE
    # =================================================

    cliente = pd.DataFrame({

        "ingreso": [ingreso],

        "deuda": [deuda],

        "moras": [moras],

        "antiguedad": [antiguedad]

    })

    # =================================================
    # PREDICCIÓN
    # =================================================

    prediccion = modelo.predict(cliente)[0]

    probabilidad = modelo.predict_proba(cliente)[0][1]

    # =================================================
    # SCORE
    # =================================================

    score = int((1 - probabilidad) * 100)

    # =================================================
    # INDICADORES FINANCIEROS
    # =================================================

    capacidad_pago = ingreso - deuda

    cuota_estimada = (
        monto_credito / cuotas
        if cuotas > 0 else monto_credito
    )

    deuda_total = deuda + cuota_estimada

    ratio_total = (
        deuda_total / ingreso
        if ingreso > 0 else 1
    )

    ratio_cuota = (
        cuota_estimada / capacidad_pago
        if capacidad_pago > 0 else 1
    )

    respaldo_financiero = (
        ahorros / deuda
        if deuda > 0 else ahorros
    )

    monto_recomendado = capacidad_pago * 12

    # =================================================
    # AJUSTES SEGÚN CONTRATO
    # =================================================

    if contrato == "Indefinido":

        monto_recomendado *= 1.2

    elif contrato == "Honorarios":

        monto_recomendado *= 0.8

    # =================================================
    # RESTRICCIÓN FINANCIERA
    # =================================================

    restriccion_cuota = False

    # Regla bancaria:
    # La cuota no puede superar
    # el 50% de la capacidad libre

    if ratio_cuota > 0.5:

        restriccion_cuota = True

    # =================================================
    # CLASIFICACIÓN
    # =================================================

    if score >= 80:

        tasa = "12%"
        riesgo_texto = "Bajo"

    elif score >= 60:

        tasa = "18%"
        riesgo_texto = "Medio"

    else:

        tasa = "25%"
        riesgo_texto = "Alto"

    # =================================================
    # RESULTADO
    # =================================================

    st.header("📈 Resultado de Evaluación")

    if prediccion == 0 and not restriccion_cuota:

        st.success("✅ Crédito aprobado")

    else:

        st.error("❌ Crédito rechazado")

    # =================================================
    # MÉTRICAS
    # =================================================

    colA, colB, colC = st.columns(3)

    with colA:

        st.metric(
            "Score Crediticio",
            score
        )

        st.metric(
            "Nivel Riesgo",
            riesgo_texto
        )

    with colB:

        st.metric(
            "Probabilidad Riesgo",
            f"{probabilidad:.2%}"
        )

        st.metric(
            "Ratio Endeudamiento",
            f"{ratio_total:.2f}"
        )

    with colC:

        st.metric(
            "Capacidad Pago",
            f"${capacidad_pago:,.0f}"
        )

        st.metric(
            "Monto Recomendado",
            f"${monto_recomendado:,.0f}"
        )

    st.metric(
        "Cuota Estimada",
        f"${cuota_estimada:,.0f}"
    )

    st.metric(
        "Ratio Cuota/Capacidad",
        f"{ratio_cuota:.2f}"
    )

    st.metric(
        "Tasa Recomendada",
        tasa
    )

    # =================================================
    # CÁLCULOS FINANCIEROS
    # =================================================

    st.subheader("🧮 Cálculos Financieros")

    st.write("### Ratio Total de Endeudamiento")

    st.write(f"""
    Ratio = (Deuda + Cuota) / Ingreso

    Ratio = ({deuda:,.0f} + {cuota_estimada:,.0f}) / {ingreso:,.0f}

    Ratio = {ratio_total:.2f}
    """)

    st.write("### Ratio Cuota / Capacidad")

    st.write(f"""
    Ratio = Cuota / Capacidad Pago

    Ratio = {cuota_estimada:,.0f} / {capacidad_pago:,.0f}

    Ratio = {ratio_cuota:.2f}
    """)

    st.write("### Capacidad de Pago")

    st.write(f"""
    Capacidad = Ingreso - Deuda

    Capacidad = {ingreso:,.0f} - {deuda:,.0f}

    Capacidad = {capacidad_pago:,.0f}
    """)

    st.write("### Cuota Estimada")

    st.write(f"""
    Cuota = Monto Crédito / Cuotas

    Cuota = {monto_credito:,.0f} / {cuotas}

    Cuota = {cuota_estimada:,.0f}
    """)

    st.write("### Respaldo Financiero")

    st.write(f"""
    Respaldo = Ahorros / Deuda

    Respaldo = {ahorros:,.0f} / {deuda:,.0f}

    Respaldo = {respaldo_financiero:.2f}
    """)

    st.write("### Score Crediticio")

    st.write(f"""
    Score = (1 - Probabilidad Riesgo) × 100

    Score = (1 - {probabilidad:.4f}) × 100

    Score = {score}
    """)

    # =================================================
    # EXPLICACIÓN FINANCIERA
    # =================================================

    st.subheader("🧠 Explicación Financiera")

    if restriccion_cuota:

        st.error("""
        La cuota mensual supera el 50%
        de la capacidad libre mensual.
        """)

    if ratio_total > 0.5:

        st.write(
            "• Alto nivel de endeudamiento."
        )

    elif ratio_total > 0.3:

        st.write(
            "• Endeudamiento moderado."
        )

    else:

        st.write(
            "• Buen nivel de endeudamiento."
        )

    if moras > 0:

        st.write(
            "• Existen moras registradas."
        )

    if antiguedad < 1:

        st.write(
            "• Baja antigüedad laboral."
        )

    if contrato == "Honorarios":

        st.write(
            "• Ingresos menos estables."
        )

    if respaldo_financiero >= 2:

        st.write(
            "• Buen respaldo financiero."
        )

    if dependientes >= 3:

        st.write(
            "• Alta carga financiera familiar."
        )

    # =================================================
    # GRÁFICO
    # =================================================

    st.subheader("📊 Indicadores Financieros")

    fig, ax = plt.subplots(figsize=(10, 5))

    categorias = [
        "Score",
        "Riesgo %",
        "Endeudamiento %",
        "Ratio Cuota"
    ]

    valores = [
        score,
        probabilidad * 100,
        ratio_total * 100,
        ratio_cuota * 100
    ]

    barras = ax.bar(
        categorias,
        valores
    )

    for barra in barras:

        altura = barra.get_height()

        ax.text(
            barra.get_x() + barra.get_width()/2,
            altura + 1,
            f"{altura:.1f}",
            ha='center'
        )

    ax.set_ylim(0, 100)

    ax.set_ylabel("Nivel")

    ax.set_title(
        "Evaluación Financiera"
    )

    st.pyplot(fig)

# =====================================================
# MATRIZ DE CONFUSIÓN
# =====================================================

st.subheader("📉 Evaluación del Modelo")

st.write(
    f"Accuracy del modelo: {accuracy:.2%}"
)

fig2, ax2 = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz
)

disp.plot(ax=ax2)

st.pyplot(fig2)

# =====================================================
# IMPORTANCIA VARIABLES
# =====================================================

st.subheader("📌 Importancia de Variables")

st.dataframe(coeficientes)

fig3, ax3 = plt.subplots()

ax3.bar(
    coeficientes["Variable"],
    coeficientes["Coeficiente"]
)

ax3.set_title(
    "Importancia de Variables"
)

st.pyplot(fig3)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🤖 Información del Modelo")

st.sidebar.success(
    "Modelo: Logistic Regression"
)

st.sidebar.write("---")

st.sidebar.info(
    f"Accuracy: {accuracy:.2%}"
)

st.sidebar.write("---")

st.sidebar.write("📌 Variables utilizadas:")

st.sidebar.write("• Ingreso")
st.sidebar.write("• Deuda")
st.sidebar.write("• Moras")
st.sidebar.write("• Antigüedad laboral")

st.sidebar.write("---")

st.sidebar.write("""
Este sistema utiliza Machine Learning
para estimar el riesgo financiero
de clientes y automatizar
la evaluación crediticia.
""")