"""Integration test fixtures for the NEC agent.

Provides session-scoped agent and LLM judge fixtures so the expensive
embedding/model warm-up only happens once per test run.

Each run writes to tests_integration/logs/<timestamp>/:
  - run.log        full logging output (INFO+)
  - results.jsonl  one JSON object per question with verdict details
  - report.txt     human-readable summary with word-wrapped failure details

Run with:  pytest tests_integration/ -v
"""

import json
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from nec_rag.agent.agent import build_nec_agent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOGS_DIR = Path(__file__).parent / "logs"

load_dotenv(PROJECT_ROOT / ".env")

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator comparing an AI agent's answer about the National \
Electrical Code (NEC) against a known correct answer.

You will receive:
- QUESTION: The original question posed to the agent.
- CORRECT ANSWER: The verified correct answer.
- AGENT RESPONSE: The agent's full response.

Determine whether the agent's response contains or is consistent with the \
correct answer. The agent may include additional context, citations, or \
explanation -- that is fine. What matters is that the core factual answer \
matches. Minor wording differences are acceptable (e.g. "6 ft 7 in" vs \
"6 feet, 7 inches"). Numerically equivalent values are acceptable.
"""


class JudgeVerdict(BaseModel):
    """Structured output schema for the LLM judge."""

    passed: bool = Field(description="True if the agent's answer matches the correct answer, False otherwise.")
    reasoning: str = Field(description="Brief explanation of why the answer was judged as correct or incorrect.")


def _ask_judge(client: AzureOpenAI, deployment: str, question: str, correct_answer: str, agent_response: str) -> JudgeVerdict:
    """Call the LLM judge and return a structured verdict via Pydantic parsing."""
    user_message = f"QUESTION:\n{question}\n\nCORRECT ANSWER:\n{correct_answer}\n\nAGENT RESPONSE:\n{agent_response}"

    response = client.beta.chat.completions.parse(
        model=deployment,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=JudgeVerdict,
    )
    verdict = response.choices[0].message.parsed
    logger.info("Judge verdict: passed=%s reasoning=%s", verdict.passed, verdict.reasoning)
    return verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_log_dir():
    """Create a timestamped log directory for this test run and attach a file handler."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = LOGS_DIR / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Attach a file handler to the root logger so all INFO+ output is captured
    log_file = log_dir / "run.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Integration test run started. Logs: %s", log_dir)
    yield log_dir

    # Cleanup: remove file handler at end of session
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def _write_report(records: list[dict], report_path: Path):
    """Write a human-readable report.txt with word-wrapped failure details."""
    passed = [r for r in records if r.get("passed")]
    failed = [r for r in records if not r.get("passed")]
    total = len(records)
    width = 88
    divider = "=" * width
    thin_divider = "-" * width

    lines: list[str] = []
    lines.append(divider)
    lines.append(f"  Integration Test Report  --  {total} questions, {len(passed)} passed, {len(failed)} failed")
    lines.append(divider)
    lines.append("")

    # Failures first (the interesting part)
    if failed:
        lines.append(f"FAILURES ({len(failed)})")
        lines.append(thin_divider)
        for i, rec in enumerate(failed, 1):
            lines.append(f"\n  [{i}] QUESTION:")
            lines.append(textwrap.fill(rec["question"], width=width, initial_indent="      ", subsequent_indent="      "))
            lines.append("\n      EXPECTED ANSWER:")
            lines.append(textwrap.fill(rec["correct_answer"], width=width, initial_indent="      ", subsequent_indent="      "))
            lines.append("\n      JUDGE REASONING:")
            lines.append(textwrap.fill(rec["reasoning"], width=width, initial_indent="      ", subsequent_indent="      "))
            lines.append("\n      AGENT RESPONSE:")
            lines.append(textwrap.fill(rec["agent_response"], width=width, initial_indent="      ", subsequent_indent="      "))
            lines.append(f"\n{thin_divider}")
    else:
        lines.append("All questions passed!")

    # Brief listing of passes
    if passed:
        lines.append(f"\nPASSED ({len(passed)})")
        lines.append(thin_divider)
        for i, rec in enumerate(passed, 1):
            q_short = textwrap.shorten(rec["question"], width=width - 10, placeholder="...")
            lines.append(f"  [{i}] {q_short}")
            lines.append(f"       Judge: {rec['reasoning']}")
        lines.append("")

    lines.append(divider)
    lines.append(f"  Score: {len(passed)}/{total}")
    lines.append(divider)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def results_writer(run_log_dir):  # pylint: disable=redefined-outer-name
    """Provide a callable that appends a result record to results.jsonl."""
    results_path = run_log_dir / "results.jsonl"

    def _write(record: dict):
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    yield _write

    # Generate summary and report at end of session
    if results_path.exists():
        raw_lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in raw_lines]
        n_passed = sum(1 for r in records if r.get("passed"))
        total = len(records)

        # Append summary to JSONL
        summary = {"summary": True, "passed": n_passed, "failed": total - n_passed, "total": total, "score": f"{n_passed}/{total}"}
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")

        # Write human-readable report
        _write_report(records, run_log_dir / "report.txt")
        logger.info("Run complete: %d/%d passed. Report: %s", n_passed, total, run_log_dir / "report.txt")


@pytest.fixture(scope="session")
def nec_agent(run_log_dir):  # pylint: disable=redefined-outer-name,unused-argument
    """Build the NEC ReAct agent once per test session (expensive warm-up)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Building NEC agent for integration tests...")
    agent = build_nec_agent()
    logger.info("NEC agent ready.")
    return agent


@pytest.fixture(scope="session")
def llm_judge():
    """Return a callable(question, correct_answer, agent_response) -> JudgeVerdict."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2-chat")
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )
    logger.info("LLM judge initialised (deployment=%s)", deployment)

    def judge(question: str, correct_answer: str, agent_response: str) -> JudgeVerdict:
        return _ask_judge(client, deployment, question, correct_answer, agent_response)

    return judge


@pytest.fixture(scope="session")
def ask_agent(nec_agent):  # pylint: disable=redefined-outer-name
    """Return a callable(question) -> str that invokes the agent and returns its final answer."""

    def _ask(question: str) -> str:
        result = nec_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    return _ask
