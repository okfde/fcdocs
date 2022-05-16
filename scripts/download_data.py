#!/usr/bin/env python3

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default=Path("data/01_raw/pdfdocs_01"), type=Path)
parser.add_argument("--limit", default=100, type=int)
parser.add_argument("--document-endpoint", required=True)
parser.add_argument("--feature-endpoint", required=True)
args = parser.parse_args()

data_path = args.data_path
data_path.mkdir(exist_ok=True)


def get_objects(
    endpoint: str, batch_size: int, limit: Optional[int] = None, offset=0
) -> list:
    remaining = limit
    objects = []
    while remaining is None or remaining > 0:
        batch_limit = batch_size
        if remaining is not None:
            batch_limit = min(remaining, batch_limit)
        req = requests.get(endpoint, params={"limit": batch_limit, "offset": offset})
        assert req.ok

        result = req.json()["results"]
        if not result:
            break

        count = len(result)
        if remaining is not None:
            remaining -= count
        offset += count
        objects += result

    return objects


def get_document(document_id: int):
    doc_url = urllib.parse.urljoin(args.document_endpoint, str(doc_id))
    document_req = requests.get(doc_url)
    assert document_req.ok
    return document_req.json()


features = get_objects(args.feature_endpoint, batch_size=10)

annotated_docs = set()
for feature in features:
    true_docs = feature["documents"]["true"]
    false_docs = feature["documents"]["false"]
    annotated_docs.update(true_docs + false_docs)

annotated_docs = list(annotated_docs)
for doc_id in tqdm.tqdm(annotated_docs[: args.limit]):
    document = get_document(doc_id)

    file_url = urllib.parse.urljoin(args.document_endpoint, document["file_url"])
    file_id = document["id"]
    file_path = data_path / f"{file_id}.pdf"
    if not file_path.exists():
        file_req = requests.get(file_url, stream=True)
        file_req.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in file_req.iter_content(chunk_size=4096):
                f.write(chunk)

    metadata = {}
    for feature in features:
        feature_name = feature["name"]
        if file_id in feature["documents"]["true"]:
            metadata[feature_name] = True
        elif file_id in feature["documents"]["false"]:
            metadata[feature_name] = False

    meta_path = data_path / f"{file_id}.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
