# Dormiu mal por causa do celular?

Uma leitura da noite a partir dos hábitos de tela antes de dormir. A pessoa preenche o formulário, o modelo estima a categoria de dívida de sono e uma sugestão diz o que vale mudar primeiro.

O modelo é o Random Forest do notebook `the-midnight-scroll.ipynb`: 200 árvores, profundidade 8, treinado no CSV do Kaggle para prever `sleep_debt_category`. A tela em Streamlit só mostra o resultado. A API em FastAPI cuida das rotas, do treino e da chamada ao modelo de linguagem.

Isto é uma demonstração. O dataset do Kaggle é simulado, e as sugestões são hábitos, não avaliação de saúde.

## Como rodar

Python 3.9 ou mais novo.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

No `.env`, coloque uma chave gratuita da [Groq](https://console.groq.com/keys):

```
LLM_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL=openai/gpt-oss-120b
```

Suba a API e a tela:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
streamlit run frontend/app.py --server.port 8501
```

Abra http://127.0.0.1:8501. A documentação das rotas fica em http://127.0.0.1:8000/docs.

| Método | Rota | O que faz |
| --- | --- | --- |
| GET | `/schema` | Campos do formulário |
| POST | `/predict` | Categoria da noite |
| POST | `/recommend` | Sugestões a partir dessa categoria |
| GET | `/health` | Se a API está no ar |

O CSV de treino já está em `data/bedtime_screentime_sleep_debt.csv`. Na primeira subida, a API treina o modelo e guarda o arquivo em `artifacts/`.
