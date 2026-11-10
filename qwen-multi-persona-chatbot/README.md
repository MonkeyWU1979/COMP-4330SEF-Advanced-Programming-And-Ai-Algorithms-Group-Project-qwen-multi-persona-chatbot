# Conversational AI Role Play: Qwen2.5-3B Multi-Persona Chatbot

COMP 4330SEF Advanced Programming and AI Algorithms — Group 01 Project

A customisable multi-persona chatbot built on **Qwen2.5-3B-Instruct** that
compares two prompt-based personalisation strategies and provides a complete
experiment pipeline for measuring persona consistency, instruction following,
response quality and response latency.

---

## 1. Features

- **Qwen2.5-3B-Instruct integration** through Hugging Face Transformers.
- **4-bit NF4 quantization** configured for 8 GB laptop GPUs (tested target:
  NVIDIA RTX 4060).
- **Two prompt strategies** as defined in the proposal:
  - **Basic Prompt** — forwards the character definition with minimal wrapping.
  - **Structured Persona Prompt** — explicit sections for personality,
    background, speaking style and behavioural rules, plus role-maintenance
    constraints.
- **Gradio web interface** with:
  - Free-text and structured persona editors.
  - Five predefined personas and multi-turn test scenarios.
  - Adjustable decoding parameters (temperature, top-p, max tokens,
    repetition penalty).
  - System prompt preview.
  - Built-in 1–5 manual rating panel that exports ratings to CSV.
- **Multi-turn chat engine** with sliding context-window management; the
  system prompt is always retained while older turns are truncated.
- **Batch evaluation runner** that executes identical scenarios under both
  prompt methods and records latency, token counts and full conversations.
- **Analysis utility** producing descriptive statistics for the final report.

---

## 2. Project Structure

```
qwen-multi-persona-chatbot/
├── app.py                         # Gradio web application entry point
├── requirements.txt               # Python dependencies
├── README.md
├── src/
│   ├── config.py                  # Global model, generation and eval config
│   ├── model_manager.py           # Qwen model loading, quantization, generation
│   ├── persona_manager.py         # Persona data model and prompt builders
│   ├── chat_engine.py             # Multi-turn dialogue and context management
│   └── evaluator.py               # Experiment runner, CSV/JSON persistence
├── experiments/
│   ├── persona_profiles.py        # Five predefined test personas
│   ├── test_scenarios.py          # Multi-turn scenarios for each persona
│   ├── run_evaluation.py          # CLI batch experiment runner
│   └── analyze_results.py         # Descriptive statistics utility
├── utils/
│   └── timing.py                  # Stopwatch helper
└── data/
    └── evaluation_results/        # CSV / JSON outputs (created at runtime)
```

---

## 3. Environment Requirements

- Python **3.10 or 3.11** (recommended for Windows `bitsandbytes` support).
- NVIDIA GPU with CUDA 12.x and at least **8 GB VRAM** when 4-bit
  quantization is enabled.
- Approximately 6 GB free disk space for the model weights on first run.
- Internet connection for the first download of Qwen2.5-3B-Instruct from
  Hugging Face Hub.

---

## 4. Installation

Open PowerShell, navigate to the project directory and run:

```powershell
# (Recommended) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Step 1: install the CUDA-enabled PyTorch build BEFORE other dependencies
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# Step 2: install the remaining dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that PyTorch can see the GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The expected output is `CUDA available: True` and the name of your RTX 4060.
If it prints `False`, the CPU build of torch is still installed; reinstall it
with the `cu128` index above.

---

## 5. Hugging Face Authentication

Authenticated requests benefit from higher rate limits and faster downloads.

1. Create a token at <https://huggingface.co/settings/tokens> (a **Read**
   permission token is sufficient).
2. Copy the template file and enter your token:

```powershell
Copy-Item .env.example .env
```

3. Open `.env` and replace the placeholder with the real token:

```text
HF_TOKEN=hf_your_real_token_here
```

The project loads `.env` automatically through `python-dotenv` and calls
`huggingface_hub.login()` on startup. The real `.env` file is excluded from
git via `.gitignore`; never share it or commit it.

> **Security note:** if a token has ever been pasted into a chat, screenshot
> or shared location, revoke it in the Hugging Face settings and generate a
> new one.

You can also authenticate without a `.env` file using an environment
variable:

```powershell
$env:HF_TOKEN = "hf_your_real_token_here"
python app.py
```


---

## 6. Running the Web Application

```powershell
python app.py
```

Then open the local URL printed in the terminal (default:
`http://127.0.0.1:7860`). The model is downloaded and loaded lazily when
the first message is sent, so the interface appears immediately.

### Typical workflow

1. Load a predefined persona or fill in the character fields manually.
2. Choose **Basic Prompt** or **Structured Persona Prompt**.
3. Click **Apply persona** (optionally clear history).
4. Chat in the right panel; latency and token counts appear under the
   conversation.
5. Open **Rate the latest response**, assign 1–5 scores and click
   **Save rating**. Ratings are appended to
   `data/evaluation_results/manual_ratings.csv`.

---

## 7. Running the Batch Experiment

Run all five personas under both prompt strategies:

```powershell
python experiments/run_evaluation.py
```

Useful options:

```powershell
# Run a subset of personas with three repeated runs
python experiments/run_evaluation.py --personas professor detective --repetitions 3

# Compare only the structured method
python experiments/run_evaluation.py --methods structured

# Use a local model path and disable quantization (requires more VRAM)
python experiments/run_evaluation.py --model-name "C:\models\Qwen2.5-3B-Instruct" --no-quantize
```

### Output files

All outputs are written to `data/evaluation_results/`:

| File | Content |
|---|---|
| `evaluation_results.csv` | Flat turn-level records: messages, latency, token counts, empty rating columns |
| `evaluation_results.json` | Nested complete conversations with experiment metadata |
| `evaluation_summary.csv` | Aggregated mean/min/max latency and token counts per condition |
| `manual_ratings.csv` | Ratings collected through the Gradio interface |

---

## 8. Analysing Results

```powershell
python experiments/analyze_results.py
```

The script prints:

- Overall response time and token statistics grouped by prompt method.
- Persona-by-persona descriptive statistics.
- Qualitative rating summaries once scores are populated.
- Aggregated manual ratings from Gradio sessions.

A flattened analysis report is saved to
`data/evaluation_results/analysis_report.csv` for direct use in the final
report.

---

## 9. Prompt Strategies

### Basic Prompt

The user's character definition is forwarded with minimal wrapping:

```text
You are <name>.
<raw character definition>
Respond to the user while staying in character as <name>.
```

### Structured Persona Prompt

The character is organised into labelled sections:

1. Character name and description.
2. Personality traits.
3. Background.
4. Speaking style.
5. Behavioral rules.
6. Explicit role-playing requirements that prohibit breaking character.

Both methods are evaluated on identical personas and scenarios so that any
difference in consistency or latency is attributable to the prompt design.

---

## 10. Evaluation Metrics

| Metric | Type | Measurement |
|---|---|---|
| Persona consistency | Qualitative, 1–5 | 1 = breaks character; 5 = fully maintains persona |
| Instruction following | Qualitative, 1–5 | 1 = ignores character instructions; 5 = follows all of them |
| Response quality | Qualitative, 1–5 | Relevance, coherence and fluency |
| Response time | Quantitative | Seconds per generated response, measured with CUDA events |

The first three dimensions are scored manually through the Gradio panel;
response time and token counts are recorded automatically.

---

## 11. Troubleshooting

### "CUDA is not available; the model will run on CPU"

This means a CPU build of PyTorch (a version ending in `+cpu`) is installed.
Reinstall the CUDA build using the same interpreter that runs the project:

```powershell
python -m pip uninstall torch -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The last command must print `True` and a CUDA version. The NVIDIA driver is
backward compatible, so a driver supporting newer CUDA also runs the CUDA
12.8 PyTorch build.

### Out-of-memory errors

- Confirm `load_in_4bit=True` (default in `src/config.py`).
- Reduce `max_new_tokens` and `max_context_length`.
- Close other GPU applications and check VRAM usage with
  `nvidia-smi`.

### `bitsandbytes` import failure on Windows

Upgrade to a recent Windows-compatible release using the same interpreter
that runs the project:

```powershell
python -m pip install -U bitsandbytes
```

### Gradio API or version errors

The interface is written for **Gradio 6.x**. If you see errors such as
`unexpected keyword argument`, install the correct version into the active
Python environment:

```powershell
python -m pip install -U "gradio>=6.0.0"
python -m pip show gradio
```

Always use `python -m pip` instead of a bare `pip` so the package is
installed into the interpreter that runs `app.py`.

### The model answers as an AI instead of the character

- Switch to the **Structured Persona Prompt**, which contains explicit
  role-maintenance rules.
- Lower the temperature slightly (0.5–0.7) to reduce drift.
- Verify the persona was applied after editing the fields.

### First run is slow

The Hugging Face cache downloads the model on first use. Subsequent runs
load weights from local disk and start much faster.

---

## 12. Reproducibility Notes

- All default hyperparameters are declared in `src/config.py`.
- Each experiment records the model identifier, timestamp, prompt method,
  repetition index and full conversation.
- Use the same `--repetitions` value across conditions when comparing
  latency.
- Keep manual rating criteria consistent across all group members; the
  rating rubric is printed directly above the Gradio rating sliders.
