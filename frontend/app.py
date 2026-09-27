import html
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

SECTIONS = [
    ("Você", ["gender", "age", "occupation_type", "chronotype"]),
    (
        "Antes de dormir",
        [
            "bedtime_phone_minutes",
            "primary_bedtime_app",
            "screen_brightness_pct",
            "blue_light_filter_active",
        ],
    ),
    (
        "No resto do dia",
        ["caffeine_post_5pm_mg", "physical_activity_min", "sleep_latency_min"],
    ),
]

HINTS = {
    "bedtime_phone_minutes": "Quanto tempo o celular ficou na mão, já na cama.",
    "screen_brightness_pct": "Quão clara a tela estava.",
    "blue_light_filter_active": "Modo noturno ou filtro de luz azul.",
    "caffeine_post_5pm_mg": "Café, chá ou energético depois das 17h.",
    "physical_activity_min": "Tempo em movimento ao longo do dia.",
    "sleep_latency_min": "Quanto tempo levou para pegar no sono.",
    "primary_bedtime_app": "O app que mais segurou você acordado.",
    "chronotype": "O horário em que o seu corpo prefere funcionar.",
}

LEVELS = [
    ("Optimal Recovery", "Em dia", "optimal"),
    ("Mild Deficit", "Leve", "mild"),
    ("Moderate Debt", "Moderada", "moderate"),
    ("Severe Sleep Debt", "Severa", "severe"),
]
LEVEL_COPY = {
    "Optimal Recovery": "A noite está em um bom lugar. Vale manter o que já funciona.",
    "Mild Deficit": "A noite escorregou um pouco. Um ajuste pequeno já muda o dia seguinte.",
    "Moderate Debt": "A tela e os hábitos da noite estão pesando no sono.",
    "Severe Sleep Debt": "Essa noite pede uma mudança clara antes de dormir.",
}
PRIORITY_LABELS = {"alta": "Comece por aqui", "media": "Depois", "baixa": "Se sobrar energia"}

st.set_page_config(page_title="Dormiu mal por causa do celular?", page_icon="🌙", layout="centered")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 2rem; max-width: 760px;}
    .hero {
        background: linear-gradient(160deg, #12182B 0%, #243B67 55%, #5C7AEA 100%);
        padding: 36px 32px 32px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {margin: 0; color: white; font-size: 40px; letter-spacing: -0.5px;}
    .hero p {margin: 10px 0 0 0; color: #E4E9FF; font-size: 17px; line-height: 1.45;}
    .verdict {
        border-radius: 24px;
        padding: 28px 28px 24px;
        color: white;
        margin: 6px 0 16px;
    }
    .verdict h2 {margin: 0; color: white; font-size: 34px;}
    .verdict p {margin: 8px 0 0; color: rgba(255,255,255,0.92); font-size: 16px;}
    .verdict.optimal {background: linear-gradient(135deg, #145C52, #3AAFA9);}
    .verdict.mild {background: linear-gradient(135deg, #8A6410, #E0A82E);}
    .verdict.moderate {background: linear-gradient(135deg, #7A3048, #C45C74);}
    .verdict.severe {background: linear-gradient(135deg, #5C1A24, #9B2335);}
    .scale {display: flex; gap: 8px; margin: 0 0 18px;}
    .scale span {
        flex: 1;
        text-align: center;
        padding: 8px 4px;
        border-radius: 999px;
        background: #F0F2F8;
        color: #5C6478;
        font-size: 13px;
    }
    .scale span.on {background: #1B2845; color: white; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def field(feature: dict):
    name = feature["name"]
    label = feature["label"]
    hint = HINTS.get(name)
    if feature["type"] == "categorical":
        labels = {option["value"]: option["label"] for option in feature["options"]}
        return st.selectbox(
            label,
            options=list(labels),
            format_func=lambda value, labels=labels: labels[value],
            key=name,
            help=hint,
        )
    if feature["type"] == "boolean":
        enabled = st.toggle(label, value=bool(feature["default"]), key=name, help=hint)
        return 1 if enabled else 0
    if feature["type"] == "integer":
        return int(
            st.slider(
                label,
                min_value=int(feature["min"]),
                max_value=int(feature["max"]),
                value=int(feature["default"]),
                step=1,
                key=name,
                help=hint,
            )
        )
    return float(
        st.slider(
            label,
            min_value=float(feature["min"]),
            max_value=float(feature["max"]),
            value=float(feature["default"]),
            step=float(feature["step"]),
            key=name,
            help=hint,
        )
    )


try:
    schema = api_get("/schema")
except requests.RequestException:
    st.markdown(
        """
        <div class="hero">
          <h1>Dormiu mal por causa do celular?</h1>
          <p>Não consegui abrir a leitura agora. Tente de novo em instantes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

features = {feature["name"]: feature for feature in schema["features"]}
st.markdown(
    """
    <div class="hero">
      <h1>Dormiu mal por causa do celular?</h1>
      <p>Conte a noite. O resultado mostra o quanto a tela pesou e o primeiro hábito a cortar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

values = {}
placed = set()
for title, names in SECTIONS:
    present = [features[name] for name in names if name in features]
    if not present:
        continue
    st.subheader(title)
    left, right = st.columns(2)
    for index, feature in enumerate(present):
        placed.add(feature["name"])
        with left if index % 2 == 0 else right:
            values[feature["name"]] = field(feature)

leftover = [feature for name, feature in features.items() if name not in placed]
if leftover:
    st.subheader("Mais um pouco")
    for feature in leftover:
        values[feature["name"]] = field(feature)

st.write("")
submitted = st.button("Ver a minha noite", type="primary", use_container_width=True)

if "night" not in st.session_state:
    st.session_state.night = None

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=values, timeout=30)
        if response.status_code == 422:
            st.error("Algum campo ficou fora do intervalo. Ajuste e tente de novo.")
            st.stop()
        response.raise_for_status()
        night = {"result": response.json(), "advice": None, "advice_error": None}
    except requests.RequestException:
        st.error("Não consegui ler a noite agora. Tente de novo.")
        st.stop()

    try:
        with st.spinner("Olhando o que dá para mudar..."):
            advice_response = requests.post(f"{API_URL}/recommend", json=values, timeout=60)
        if advice_response.status_code >= 400:
            detail = advice_response.json().get("detail", {})
            message = detail.get("message") if isinstance(detail, dict) else None
            night["advice_error"] = message or "As sugestões não chegaram desta vez."
        else:
            night["advice"] = advice_response.json()["recommendations"]
    except requests.RequestException:
        night["advice_error"] = "A leitura ficou pronta, mas as sugestões não chegaram."
    st.session_state.night = night

night = st.session_state.night
if night:
    result = night["result"]
    category = result["sleep_debt_category"]
    tone = next((item[2] for item in LEVELS if item[0] == category), "moderate")
    st.markdown(
        f"""
        <div class="verdict {tone}">
          <h2>{html.escape(result["label"])}</h2>
          <p>{html.escape(LEVEL_COPY.get(category, "Uma leitura da sua noite."))}</p>
        </div>
        <div class="scale">
          {"".join(
              f'<span class="{"on" if key == category else ""}">{html.escape(short)}</span>'
              for key, short, _tone in LEVELS
          )}
        </div>
        """,
        unsafe_allow_html=True,
    )

    advice = night["advice"]
    if advice:
        st.write(advice["summary"])
        st.subheader("O que mudar")
        for action in advice["actions"]:
            with st.container(border=True):
                st.markdown(f"**{action['title']}**")
                st.caption(PRIORITY_LABELS.get(action["priority"], "Depois"))
                st.write(action["detail"])
        st.caption(advice["note"])
    elif night["advice_error"]:
        st.info(night["advice_error"])

st.caption(
    "Sugestão de hábito feita por um modelo de demonstração. Não é avaliação de saúde."
)
