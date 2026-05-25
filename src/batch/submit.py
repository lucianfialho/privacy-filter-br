"""cmd_submit: upload batch JSONL to OpenAI Files API + create batch job."""
from __future__ import annotations

from pathlib import Path

import openai


def cmd_submit(args) -> None:
    """Upload JSONL and create a batch job."""
    client = openai.OpenAI()
    print(f"Uploading {args.input} ...")
    with open(args.input, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  file id: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Created batch: {batch.id}")
    print(f"  status: {batch.status}")

    out_id = Path(args.input).with_name("batch_id.txt")
    out_id.write_text(batch.id)
    print(f"Saved batch id to {out_id}")
