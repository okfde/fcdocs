#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import requests

BASE_URL = "https://fragdenstaat.de/api/v1/attachment/?filetype=application/pdf&offset={offset}&limit={limit}"

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default=Path("data/01_raw/pdfdocs_01"), type=Path)
parser.add_argument("--batch_size", default=50, type=int)
parser.add_argument("--offset", default=0, type=int)
parser.add_argument("--limit", default=100, type=int)
args = parser.parse_args()

data_path = args.data_path
data_path.mkdir(exist_ok=True)

remaining = args.limit
offset = args.offset
session = requests.Session()
while remaining > 0:
    url = BASE_URL.format(offset=offset, limit=min(remaining, args.limit))
    req = session.get(url)
    data = req.json()["objects"]
    count = len(data)
    remaining -= count
    offset += count

    for attachment in data:
        file_url = attachment["file_url"]
        file_id = attachment["id"]
        file_path = data_path / f"{file_id}.pdf"
        if not file_path.exists():
            print(f"Downloading to {file_path}")
            file_req = session.get(file_url, stream=True)
            file_req.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in file_req.iter_content(chunk_size=4096):
                    f.write(chunk)

        meta_path = data_path / f"{file_id}.json"
        if not meta_path.exists():
            print(f"Writing meta for {file_id}")
            with open(meta_path, "w") as f:
                json.dump(attachment, f)
