# my-llm-project

Tiny local LLM demo: FastAPI backend + simple HTML frontend.

## Requirements
- Python 3.11
- macOS/Linux/Windows
- (Optional) GPU not required

## Setup (local)

```bash
# 1) clone
git clone https://github.com/<YOUR_USERNAME>/my-llm-project.git
cd my-llm-project

# 2) python env
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) deps
pip install --upgrade pip
pip install -r requirements.txt

# 4) config
cp .env.example .env
# edit .env if you want different model/settings
