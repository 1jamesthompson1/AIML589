"""Download responses from the configured Qualtrics survey.

Uses QUALTRICS_API_KEY and QUALTRICS_SURVEY_ID from the repo-root .env.
The default outputs are survey/output/responses.csv and
survey/output/responses-question-ids.csv (overwritten on reruns).

    uv run survey/download_results.py
    uv run survey/download_results.py --format json
    uv run survey/download_results.py --output survey/output/responses.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DIR = Path(__file__).resolve().parent
REPO = DIR.parent
HOST = "vuw.yul1.qualtrics.com"

load_dotenv(REPO / ".env")


def api_request(
    method: str,
    path: str,
    api_key: str,
    body: dict | None = None,
    *,
    raw: bool = False,
) -> dict | bytes:
    """Make one Qualtrics API request, returning JSON or the raw file bytes.

    ``path`` may be an API path or an absolute ``nextPage`` URL. The API's
    pagination links are absolute URLs, so both forms are accepted.
    """
    url = path if path.startswith(("http://", "https://")) else f"https://{HOST}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Accept": "application/json",
            "X-API-TOKEN": api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read() if raw else json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{method} {path}: HTTP {error.code}: {detail}") from error


def download_export(
    survey_id: str,
    api_key: str,
    file_format: str,
    timeout: int,
) -> bytes:
    """Start a response export, wait for it, and download its ZIP archive."""
    base = f"/API/v3/surveys/{survey_id}/export-responses"
    started = api_request(
        "POST",
        base,
        api_key,
        {
            "format": file_format,
            "useLabels": True,
        },
    )
    progress_id = started["result"]["progressId"]
    deadline = time.monotonic() + timeout
    last_progress = None

    while True:
        result = api_request("GET", f"{base}/{progress_id}", api_key)["result"]
        if result.get("fileId"):
            file_id = result["fileId"]
            break
        status = result.get("status", "unknown")
        if status in {"failed", "completeWithErrors"}:
            raise RuntimeError(f"Qualtrics export did not complete: {result}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Qualtrics export was still {status!r} after {timeout} seconds"
            )
        progress = result.get("percentComplete")
        if progress != last_progress:
            print(f"Export progress: {progress}% ({status})", flush=True)
            last_progress = progress
        time.sleep(2)

    return api_request("GET", f"{base}/{file_id}/file", api_key, raw=True)


def extract_response_file(archive_bytes: bytes, file_format: str, output: Path) -> None:
    """Extract the one CSV/JSON response file from Qualtrics' ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(f".{file_format}")
        ]
        if len(names) != 1:
            raise RuntimeError(
                f"Expected one .{file_format} file in the export; found {names}"
            )
        contents = archive.read(names[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(contents)


def all_questions(survey_id: str, api_key: str) -> list[dict]:
    """Every question from the survey-definitions endpoint, across pages.

    Qualtrics returns a ``nextPage`` URL for collection requests. Follow it
    until it is empty and guard against a repeated URL so a malformed
    response cannot loop forever.
    """
    path = f"/API/v3/survey-definitions/{survey_id}/questions"
    seen = set()
    questions = []
    while path:
        if path in seen:
            raise RuntimeError(f"question pagination loop at {path!r}")
        seen.add(path)
        result = api_request("GET", path, api_key)["result"]
        questions.extend(result.get("elements") or [])
        path = result.get("nextPage")
    return questions


def download_question_map(survey_id: str, api_key: str, output: Path) -> None:
    """Write Qualtrics question IDs alongside their complete DataExportTags.

    The third CSV header row identifies columns by QID even when Qualtrics
    truncates the DataExportTag in its first header row. This map recovers the
    full comparison ID and suffix (for example, ``_pref`` or ``_aiuse``). All
    question pages are read, so the map is complete for large surveys.
    """
    questions = all_questions(survey_id, api_key)
    fields = (
        "question_id",
        "data_export_tag",
        "question_type",
        "selector",
        "description",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        seen_ids = set()
        for question in questions:
            question_id = question.get("QuestionID", "")
            if question_id and question_id in seen_ids:
                continue
            if question_id:
                seen_ids.add(question_id)
            writer.writerow(
                {
                    "question_id": question.get("QuestionID", ""),
                    "data_export_tag": question.get("DataExportTag", ""),
                    "question_type": question.get("QuestionType", ""),
                    "selector": question.get("Selector", ""),
                    "description": question.get("QuestionDescription", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey-id",
        default=os.environ.get("QUALTRICS_SURVEY_ID"),
        help="Qualtrics survey ID (default: QUALTRICS_SURVEY_ID from .env).",
    )
    parser.add_argument(
        "--format", choices=("csv", "json"), default="csv", dest="file_format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: survey/output/responses.<format>).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for the export (default: 600).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("QUALTRICS_API_KEY")
    if not api_key:
        parser.error("QUALTRICS_API_KEY is not set in the environment or .env")
    if not args.survey_id:
        parser.error("set QUALTRICS_SURVEY_ID in .env or pass --survey-id")

    output = args.output or DIR / "output" / f"responses.{args.file_format}"
    question_map = output.with_name(f"{output.stem}-question-ids.csv")
    archive = download_export(args.survey_id, api_key, args.file_format, args.timeout)
    extract_response_file(archive, args.file_format, output)
    print(f"Downloaded responses to {output}")
    download_question_map(args.survey_id, api_key, question_map)
    print(f"Downloaded question ID mapping to {question_map}")


if __name__ == "__main__":
    main()
