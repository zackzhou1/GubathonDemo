# Gubathon Demo

## Setup Instructions

### 1. Create and Activate a Virtual Environment

```sh
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

---

### 2. Install Requirements

```sh
pip install -r requirements.txt
```

---

### 3. Create the keyfile Configuration Files

```yaml
# keyfile.yaml
open_ai_api_key: your-openai-api-key-here
```

The `keyfile.yaml` file is listed in `.gitignore` by default.

---

### 4. Run the Application

```sh
python main.py
```
