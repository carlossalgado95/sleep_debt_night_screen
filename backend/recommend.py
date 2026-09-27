import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """Você sugere hábitos para reduzir a dívida de sono a partir da previsão de um modelo.
Responda apenas com um JSON com:
- summary: um parágrafo curto, em português, sobre o que a previsão indica
- actions: lista de 3 a 5 objetos com title, detail e priority
priority deve ser "alta", "media" ou "baixa".
Priorize mudanças nos hábitos enviados, sobretudo nas variáveis mais importantes do modelo e nos valores piores que a mediana do treino.
Não prescreva medicamento, dose, suplemento ou diagnóstico.
Se a categoria for dívida severa, deixe claro que a orientação é de hábito e que sono ruim persistente precisa de avaliação profissional.
"""


class RecommendationError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def recommendation_context(payload: dict, prediction: dict, bundle: dict) -> dict:
    habits = []
    for feature in bundle["schema"]:
        name = feature["name"]
        value = payload[name]
        entry = {"variavel": feature["label"]}
        if feature["type"] == "categorical":
            entry["valor"] = next(
                option["label"]
                for option in feature["options"]
                if option["value"] == value
            )
        elif feature["type"] == "boolean":
            entry["valor"] = "ativo" if int(value) == 1 else "desligado"
        else:
            entry["valor"] = value
            entry["mediana_do_treino"] = feature["default"]
        habits.append(entry)
    return {
        "previsao": prediction["label"],
        "categoria": prediction["sleep_debt_category"],
        "probabilidades": prediction["probabilities"],
        "habitos": habits,
        "importancia_das_variaveis": bundle["metrics"]["feature_importance"],
    }


def parse_advice(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecommendationError("A API devolveu um texto que não é JSON.") from exc

    summary = str(data.get("summary", "")).strip()
    actions = data.get("actions")
    if not summary or not isinstance(actions, list):
        raise RecommendationError("A API não devolveu recomendações utilizáveis.")

    cleaned = []
    for action in actions[:5]:
        if not isinstance(action, dict):
            continue
        title = str(action.get("title", "")).strip()
        detail = str(action.get("detail", "")).strip()
        priority = str(action.get("priority", "media")).strip().lower()
        if priority not in {"alta", "media", "baixa"}:
            priority = "media"
        if title and detail:
            cleaned.append(
                {"title": title, "detail": detail, "priority": priority}
            )
    if not cleaned:
        raise RecommendationError("A API não devolveu recomendações utilizáveis.")
    return {
        "summary": summary,
        "actions": cleaned,
        "note": (
            "Sugestões de hábito a partir da previsão do modelo. "
            "Não substituem avaliação de um profissional de saúde."
        ),
    }


def _env_file() -> dict:
    values = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def llm_settings() -> tuple[str, str, str]:
    file_env = _env_file()
    api_key = file_env.get("LLM_API_KEY") or os.getenv("LLM_API_KEY", "").strip()
    api_key = api_key or file_env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RecommendationError(
            "Defina LLM_API_KEY no arquivo .env. Para testar sem custo, use uma chave da Groq.",
            status_code=503,
        )
    url = file_env.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_OPENAI_URL
    model = (
        file_env.get("LLM_MODEL")
        or os.getenv("LLM_MODEL", "").strip()
        or file_env.get("OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    return api_key, url, model


def build_recommendations(context: dict) -> dict:
    api_key, url, model = llm_settings()
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False),
                    },
                ],
            },
            timeout=45,
        )
    except requests.RequestException as exc:
        raise RecommendationError("Não foi possível falar com a API do modelo.") from exc

    if response.status_code >= 400:
        message = "A API do modelo recusou o pedido."
        try:
            error = response.json().get("error", {}).get("message")
        except ValueError:
            error = None
        if error:
            message = f"{message} {error}"
        raise RecommendationError(message)

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RecommendationError("A API do modelo devolveu uma resposta inesperada.") from exc
    return parse_advice(content)
