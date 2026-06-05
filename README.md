# Math Mistake Diagnosis Toolkit

A compact, public-GitHub-friendly version of a personal math mistake analysis pipeline.

This project turns an annotated math PDF into a Markdown review report by using a Gemini/Vertex model to:

1. extract handwritten mistake problems from the PDF,
2. diagnose the student's reasoning error,
3. generate quick review cards,
4. group recurring error patterns,
5. save one printable Markdown report.

The repository intentionally keeps the core logic in one Python file:

```text
math_mistake_diagnosis.py
```

No API key, local Windows path, GCS bucket name, checkpoint file, service-account JSON file, or private PDF is included.

## Project structure

```text
.
├── math_mistake_diagnosis.py   # one-file core pipeline
├── requirements.txt            # Python dependencies
├── .env.example                # environment variable template
├── .gitignore                  # files that should not be committed
└── SECURITY.md                 # security notes for API keys and PDFs
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Then edit `.env`:

```text
GEMINI_API_KEY=replace_with_your_key
GEMINI_MODEL=gemini-2.5-pro
```

Do not commit `.env`.

## Run

Example:

```bash
python math_mistake_diagnosis.py ^
  --pdf-uri gs://your-bucket/your-annotated-math.pdf ^
  --title "880 Advanced Part 6" ^
  --pages 1-30 ^
  --output-dir outputs
```

The script creates:

```text
outputs/diagnosis_checkpoint.json
outputs/mistake_diagnosis_report.md
```

If the run is interrupted or reaches the budget limit, run the same command again. It will continue from the checkpoint.

## Useful options

```bash
python math_mistake_diagnosis.py --help
```

Common options:

```text
--pages 1-30             scan a specific page range
--chunk-size 8           extract questions in smaller page chunks
--budget-usd 15          stop after estimated cost exceeds this amount
--no-flashcards          skip review cards and error-pattern index
--model MODEL_NAME       override the model name
```

## What this does not include

This repository does not include:

- private PDFs,
- generated reports,
- checkpoint JSON files,
- debug outputs,
- `.env`,
- Google service-account JSON files,
- API keys,
- local paths from the author's computer.

## Why this version exists

The original development scripts were useful for experimentation, but they repeated the same pieces many times: model calls, retry logic, cost tracking, checkpointing, JSON cleanup, Markdown assembly, and prompt templates. This version keeps the best parts and removes the clutter.

## License

No license is included yet. Add one later if you want other people to reuse the code formally.
