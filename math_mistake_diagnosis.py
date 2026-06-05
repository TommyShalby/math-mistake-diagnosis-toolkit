#!/usr/bin/env python3
"""
Math Mistake Diagnosis Toolkit
==============================

A compact, GitHub-friendly version of the original PyCharm scripts.
It reads an annotated math PDF from a GCS URI, asks a Gemini/Vertex model to:

1. Extract mistake questions and handwriting notes.
2. Diagnose each mistake in detail.
3. Generate quick review cards.
4. Build an error-pattern index.
5. Assemble one Markdown report.

No API key, local path, bucket name, or credential file is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# ----------------------------- small utilities -----------------------------

RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"

_print_lock = threading.Lock()


def cprint(message: str = "", end: str = "\n") -> None:
    with _print_lock:
        sys.stdout.write(str(message) + end)
        sys.stdout.flush()


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so the project works even without python-dotenv."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|markdown|md)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text, flags=re.I)
    return text.strip()


def safe_parse_json(raw: str) -> Any:
    """Parse JSON returned by LLMs, with conservative cleanup."""
    text = strip_code_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the largest likely JSON array/object.
    starts = [i for i, ch in enumerate(text) if ch in "[{" ]
    ends = [i for i, ch in enumerate(text) if ch in "]}" ]
    if starts and ends and ends[-1] > starts[0]:
        candidate = text[starts[0] : ends[-1] + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)

    raise ValueError("Could not parse JSON response. First 500 chars:\n" + text[:500])


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def make_key(question: Dict[str, Any], fallback_index: int = 0) -> str:
    page = str(question.get("page", "unknown")).replace("/", "-")
    number = str(question.get("question_number", fallback_index)).replace("/", "-")
    return f"p{page}_{number}"


def parse_page_range(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    value = value.strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", value)
    if not match:
        raise ValueError("--pages must look like 1-30")
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end < start:
        raise ValueError("Invalid page range")
    return start, end


def iter_chunks(page_range: Optional[Tuple[int, int]], chunk_size: int) -> Iterable[Tuple[Optional[int], Optional[int]]]:
    if not page_range:
        yield None, None
        return
    start, end = page_range
    current = start
    while current <= end:
        yield current, min(current + chunk_size - 1, end)
        current += chunk_size


# ----------------------------- cost tracking ------------------------------


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    budget_usd: float
    price_input_per_million: float = 2.00
    price_output_per_million: float = 12.00
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.price_input_per_million
            + self.output_tokens / 1_000_000 * self.price_output_per_million
        )

    def record(self, prompt_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(prompt_tokens, 0)
        self.output_tokens += max(output_tokens, 0)
        self.calls += 1
        if self.cost() > self.budget_usd:
            raise BudgetExceeded(f"Budget exceeded: ${self.cost():.3f} / ${self.budget_usd:.2f}")

    def status(self) -> str:
        return (
            f"${self.cost():.3f}/${self.budget_usd:.0f} "
            f"in={self.input_tokens/1000:.1f}K "
            f"out={self.output_tokens/1000:.1f}K calls={self.calls}"
        )


# ----------------------------- Gemini client ------------------------------


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


@dataclass
class GeminiClient:
    api_key: str
    model: str
    tracker: CostTracker
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40

    def endpoint(self) -> str:
        return f"https://aiplatform.googleapis.com/v1/publishers/google/models/{self.model}:generateContent?key={self.api_key}"

    @staticmethod
    def _backoff(attempt: int, base: float = 6.0) -> float:
        return base * (2**attempt) + random.uniform(0, base * 0.5)

    def generate(
        self,
        prompt: str,
        *,
        file_uri: Optional[str] = None,
        max_output_tokens: int = 8192,
        force_json: bool = False,
        step: str = "",
        retries: int = 4,
        timeout: int = 600,
    ) -> str:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        if file_uri:
            parts.append({"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}})

        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_output_tokens,
            "temperature": self.temperature,
            "topP": self.top_p,
            "topK": self.top_k,
        }
        if force_json:
            generation_config["response_mime_type"] = "application/json"

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": SAFETY_SETTINGS,
        }

        last_error = "unknown error"
        for attempt in range(retries):
            try:
                response = requests.post(self.endpoint(), json=payload, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        last_error = f"no candidates: {data.get('promptFeedback')}"
                        time.sleep(self._backoff(attempt))
                        continue
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if not content_parts or "text" not in content_parts[0]:
                        last_error = f"empty text, finish={candidates[0].get('finishReason')}"
                        time.sleep(self._backoff(attempt))
                        continue

                    text = content_parts[0]["text"]
                    usage = data.get("usageMetadata", {})
                    prompt_tokens = usage.get("promptTokenCount") or int(len(prompt) * 1.5)
                    output_tokens = usage.get("candidatesTokenCount") or int(len(text) * 1.5)
                    self.tracker.record(prompt_tokens, output_tokens)
                    return text

                if response.status_code == 429:
                    wait = self._backoff(attempt, base=20)
                    cprint(f"  {YELLOW}rate limited during {step}; sleeping {wait:.1f}s{RESET}")
                    time.sleep(wait)
                    continue

                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                time.sleep(self._backoff(attempt))
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - CLI tool should show clean retry behavior
                last_error = str(exc)
                time.sleep(self._backoff(attempt))

        raise RuntimeError(f"Gemini call failed after {retries} retries during {step}: {last_error}")


# ------------------------------- prompts ----------------------------------


def extraction_prompt(title: str, page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start is None:
        page_text = "all pages"
    else:
        page_text = f"pages {page_start} to {page_end}"

    return f"""
You are reading an annotated Chinese graduate-entrance math PDF: {title}.
Task: scan {page_text} and extract every problem with clear handwritten marks, especially black-pen mistakes, red-pen corrections, crosses, erasures, or partial attempts.

Return a pure JSON array. Each item must have exactly these fields:
- page: integer
- question_number: string
- original_question_summary: short Chinese summary of the printed question; do not use LaTeX backslashes
- subtopics: Chinese comma-separated core math topics
- student_black_pen: Chinese summary of the student's black-pen attempt, at most 3 sentences
- student_red_pen: Chinese summary of red-pen correction marks, at most 3 sentences
- completeness: one of complete, partial, empty, unknown

Hard rules:
1. Output JSON only. No Markdown fence. No explanation.
2. Do not use LaTeX backslashes in JSON strings. Use Chinese description or Unicode math symbols if needed.
3. Keep each field concise but useful.
4. If a problem has no handwriting, skip it.
""".strip()


def diagnosis_prompt(title: str, question: Dict[str, Any]) -> str:
    return f"""
You are a strict but helpful kaoyan math examiner and mistake-diagnosis coach.
The PDF is attached. Locate this exact problem in the PDF before writing the diagnosis.

PDF title: {title}
Page: {question.get('page')}
Question number: {question.get('question_number')}
Known topics: {question.get('subtopics', '')}
Printed question summary: {question.get('original_question_summary', '')}
Black-pen attempt: {question.get('student_black_pen', '')}
Red-pen marks: {question.get('student_red_pen', '')}
Completeness: {question.get('completeness', '')}

Write the output in Chinese Markdown with this structure:

### Question {question.get('question_number')} - page {question.get('page')}

**Original problem:**
Quote or reconstruct the printed problem as accurately as possible. Use standard Markdown math when needed.

**Core topics:**
List the precise topics.

**Black-pen thinking reconstruction:**
Explain what the student was probably thinking, step by step. Focus on the actual handwriting.

**Mistake root cause:**
Identify the exact step where the reasoning went wrong. Name the cognitive trap.

**Correct solution:**
Give a complete, exam-ready solution. If useful, include a second faster method.

**Exam warning:**
Give 3-5 short anti-mistake rules and one memorable sentence.
""".strip()


def flashcard_prompt(question: Dict[str, Any], diagnosis_md: str) -> str:
    return f"""
Make a 3-second review card in Chinese for this diagnosed math mistake.

Question metadata:
{json.dumps(question, ensure_ascii=False, indent=2)}

Diagnosis:
{diagnosis_md[:5000]}

Return pure JSON with exactly these fields:
- error_pattern
- deadly_trap
- correct_path_keyword
- answer_oneliner

Keep each value short, sharp, and suitable for final exam review.
""".strip()


def error_index_prompt(cards: Dict[str, Dict[str, str]]) -> str:
    rows = []
    for key, card in cards.items():
        rows.append(
            f"{key} | {card.get('error_pattern', '?')} | "
            f"{card.get('deadly_trap', '?')} | {card.get('correct_path_keyword', '?')}"
        )
    return f"""
Build a Chinese Markdown error-pattern index from these quick review cards.
Group all problems into 6-12 recurring mistake patterns.

Rows: key | error pattern | deadly trap | correct path keyword
{chr(10).join(rows)}

Output structure:
## Error Pattern Index

### Pattern 1: <Chinese pattern name>
**Symptom:** ...
**Reverse reflex:** ...
**Problems:**
- key: one-line reason

Cover every key. Output Markdown only.
""".strip()


# ------------------------------ checkpoint --------------------------------


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "questions": [],
                "chunks_done": [],
                "diagnoses": {},
                "flashcards": {},
                "error_index": "",
                "meta": {},
            }

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def mark_chunk(self, chunk_key: str, questions: List[Dict[str, Any]]) -> None:
        done = set(self.data.get("chunks_done", []))
        done.add(chunk_key)
        self.data["chunks_done"] = sorted(done)
        self.data[f"chunk_{chunk_key}"] = questions
        self.save()

    def set_questions(self, questions: List[Dict[str, Any]]) -> None:
        self.data["questions"] = questions
        self.save()

    def set_diagnosis(self, key: str, markdown: str) -> None:
        self.data.setdefault("diagnoses", {})[key] = markdown
        self.save()

    def set_flashcard(self, key: str, card: Dict[str, str]) -> None:
        self.data.setdefault("flashcards", {})[key] = card
        self.save()

    def set_error_index(self, markdown: str) -> None:
        self.data["error_index"] = markdown
        self.save()


# ------------------------------- pipeline ---------------------------------


def deduplicate_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for index, question in enumerate(questions):
        key = (question.get("page"), str(question.get("question_number", index)).strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)
    return unique


def run_extraction(
    client: GeminiClient,
    ckpt: Checkpoint,
    *,
    title: str,
    pdf_uri: str,
    pages: Optional[Tuple[int, int]],
    chunk_size: int,
    max_output_tokens: int,
) -> List[Dict[str, Any]]:
    if ckpt.data.get("questions"):
        questions = ckpt.data["questions"]
        cprint(f"{GREEN}Using {len(questions)} questions from checkpoint.{RESET}")
        return questions

    all_questions: List[Dict[str, Any]] = []
    done = set(ckpt.data.get("chunks_done", []))

    for page_start, page_end in iter_chunks(pages, chunk_size):
        chunk_key = "all" if page_start is None else f"{page_start}-{page_end}"
        if chunk_key in done:
            chunk_questions = ckpt.data.get(f"chunk_{chunk_key}", [])
            cprint(f"{GREEN}Skipping extracted chunk {chunk_key} ({len(chunk_questions)} questions).{RESET}")
            all_questions.extend(chunk_questions)
            continue

        cprint(f"{CYAN}Extracting chunk {chunk_key}...{RESET} {DIM}{client.tracker.status()}{RESET}")
        raw = client.generate(
            extraction_prompt(title, page_start, page_end),
            file_uri=pdf_uri,
            max_output_tokens=max_output_tokens,
            force_json=True,
            step=f"extract {chunk_key}",
        )
        parsed = safe_parse_json(raw)
        if not isinstance(parsed, list):
            raise ValueError("Extraction did not return a JSON array")
        ckpt.mark_chunk(chunk_key, parsed)
        all_questions.extend(parsed)
        time.sleep(1)

    questions = deduplicate_questions(all_questions)
    ckpt.set_questions(questions)
    cprint(f"{GREEN}Extracted {len(questions)} unique questions.{RESET}")
    return questions


def run_diagnosis(
    client: GeminiClient,
    ckpt: Checkpoint,
    *,
    title: str,
    pdf_uri: str,
    questions: List[Dict[str, Any]],
    max_output_tokens: int,
) -> None:
    diagnoses = ckpt.data.setdefault("diagnoses", {})
    total = len(questions)

    for index, question in enumerate(questions, 1):
        key = make_key(question, index)
        if key in diagnoses:
            continue
        cprint(f"{CYAN}[{index}/{total}] Diagnosing {key}...{RESET} {DIM}{client.tracker.status()}{RESET}")
        try:
            markdown = client.generate(
                diagnosis_prompt(title, question),
                file_uri=pdf_uri,
                max_output_tokens=max_output_tokens,
                step=f"diagnose {key}",
            )
            ckpt.set_diagnosis(key, strip_code_fence(markdown))
        except BudgetExceeded:
            raise
        except Exception as exc:  # keep progress for long runs
            cprint(f"{YELLOW}Failed {key}: {exc}{RESET}")
        time.sleep(1)


def run_flashcards(
    client: GeminiClient,
    ckpt: Checkpoint,
    *,
    questions: List[Dict[str, Any]],
    max_output_tokens: int,
) -> None:
    diagnoses = ckpt.data.get("diagnoses", {})
    flashcards = ckpt.data.setdefault("flashcards", {})

    for index, question in enumerate(questions, 1):
        key = make_key(question, index)
        if key in flashcards or key not in diagnoses:
            continue
        cprint(f"{CYAN}Building quick card for {key}...{RESET} {DIM}{client.tracker.status()}{RESET}")
        try:
            raw = client.generate(
                flashcard_prompt(question, diagnoses[key]),
                max_output_tokens=max_output_tokens,
                force_json=True,
                step=f"flashcard {key}",
            )
            card = safe_parse_json(raw)
            if not isinstance(card, dict):
                raise ValueError("flashcard response was not a JSON object")
            ckpt.set_flashcard(key, card)
        except BudgetExceeded:
            raise
        except Exception as exc:
            cprint(f"{YELLOW}Failed card {key}: {exc}{RESET}")
        time.sleep(0.5)


def run_error_index(client: GeminiClient, ckpt: Checkpoint, *, max_output_tokens: int) -> None:
    cards = ckpt.data.get("flashcards", {})
    if not cards:
        return
    if ckpt.data.get("error_index"):
        return
    cprint(f"{CYAN}Building error pattern index...{RESET} {DIM}{client.tracker.status()}{RESET}")
    markdown = client.generate(
        error_index_prompt(cards),
        max_output_tokens=max_output_tokens,
        step="error index",
    )
    ckpt.set_error_index(strip_code_fence(markdown))


def assemble_report(output_path: Path, title: str, pdf_uri: str, questions: List[Dict[str, Any]], ckpt: Checkpoint, tracker: CostTracker) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnoses = ckpt.data.get("diagnoses", {})
    flashcards = ckpt.data.get("flashcards", {})
    lines: List[str] = []
    lines.append(f"# {title} - Math Mistake Diagnosis Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Source PDF: `{pdf_uri}`")
    lines.append(f"> Questions: {len(questions)}")
    lines.append(f"> Completed diagnoses: {len(diagnoses)}")
    lines.append(f"> Estimated model cost: {tracker.status()}")
    lines.append("")

    if ckpt.data.get("error_index"):
        lines.append(ckpt.data["error_index"])
        lines.append("")
        lines.append("---")
        lines.append("")

    for index, question in enumerate(questions, 1):
        key = make_key(question, index)
        lines.append(f"## {key}")
        lines.append("")
        if key in flashcards:
            card = flashcards[key]
            lines.append("### Quick Review Card")
            lines.append(f"- Error pattern: {card.get('error_pattern', '?')}")
            lines.append(f"- Deadly trap: {card.get('deadly_trap', '?')}")
            lines.append(f"- Correct path: {card.get('correct_path_keyword', '?')}")
            lines.append(f"- One-line answer: {card.get('answer_oneliner', '?')}")
            lines.append("")
        if key in diagnoses:
            lines.append(diagnoses[key])
        else:
            lines.append("_Diagnosis not completed yet. Re-run the command to continue from checkpoint._")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    cprint(f"{GREEN}Report saved to {output_path}{RESET}")


# ---------------------------------- CLI ------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a math mistake diagnosis report from an annotated PDF.")
    parser.add_argument("--pdf-uri", required=True, help="GCS PDF URI, for example gs://your-bucket/file.pdf")
    parser.add_argument("--title", default="Math Mistake Diagnosis", help="Report title")
    parser.add_argument("--pages", default=None, help="Page range, for example 1-30. Omit to scan all pages.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Pages per extraction chunk")
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoint and report")
    parser.add_argument("--report-name", default="mistake_diagnosis_report.md", help="Markdown report filename")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"), help="Gemini/Vertex model name")
    parser.add_argument("--budget-usd", type=float, default=15.0, help="Stop when estimated cost exceeds this amount")
    parser.add_argument("--extract-max-output", type=int, default=16384)
    parser.add_argument("--diagnose-max-output", type=int, default=8192)
    parser.add_argument("--card-max-output", type=int, default=2048)
    parser.add_argument("--index-max-output", type=int, default=8192)
    parser.add_argument("--no-flashcards", action="store_true", help="Skip quick cards and error index")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        cprint(f"{YELLOW}Missing GEMINI_API_KEY. Create a .env file or set an environment variable.{RESET}")
        return 2

    pages = parse_page_range(args.pages)
    output_dir = Path(args.output_dir)
    ckpt = Checkpoint(output_dir / "diagnosis_checkpoint.json")
    tracker = CostTracker(args.budget_usd)
    client = GeminiClient(api_key=api_key, model=args.model, tracker=tracker)

    ckpt.data.setdefault("meta", {})
    ckpt.data["meta"].update(
        {
            "title": args.title,
            "pdf_uri": args.pdf_uri,
            "model": args.model,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    ckpt.save()

    try:
        questions = run_extraction(
            client,
            ckpt,
            title=args.title,
            pdf_uri=args.pdf_uri,
            pages=pages,
            chunk_size=args.chunk_size,
            max_output_tokens=args.extract_max_output,
        )
        run_diagnosis(
            client,
            ckpt,
            title=args.title,
            pdf_uri=args.pdf_uri,
            questions=questions,
            max_output_tokens=args.diagnose_max_output,
        )
        if not args.no_flashcards:
            run_flashcards(client, ckpt, questions=questions, max_output_tokens=args.card_max_output)
            run_error_index(client, ckpt, max_output_tokens=args.index_max_output)
        assemble_report(output_dir / args.report_name, args.title, args.pdf_uri, questions, ckpt, tracker)
        cprint(f"{GREEN}Done. {tracker.status()}{RESET}")
        return 0
    except BudgetExceeded as exc:
        cprint(f"{YELLOW}{exc}{RESET}")
        assemble_report(output_dir / args.report_name, args.title, args.pdf_uri, ckpt.data.get("questions", []), ckpt, tracker)
        return 3
    except KeyboardInterrupt:
        cprint(f"{YELLOW}Interrupted. Progress is saved in the checkpoint.{RESET}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
