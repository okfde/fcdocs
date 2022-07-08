#!/usr/bin/env python3

import argparse
import json
import urllib.parse
import warnings
from pathlib import Path
from typing import Optional

import requests
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default=Path("data/01_raw/pdfdocs_01"), type=Path)
parser.add_argument("--limit", default=None, type=int)
parser.add_argument("--document-endpoint", required=True)
parser.add_argument("--feature-endpoint", required=True)
args = parser.parse_args()

data_path = args.data_path
data_path.mkdir(exist_ok=True)


def get_objects(
    endpoint: str,
    batch_size: int,
    limit: Optional[int] = None,
    offset: int = 0,
    key: str = "objects",
) -> list:
    remaining = limit
    objects = []
    while remaining is None or remaining > 0:
        batch_limit = batch_size
        if remaining is not None:
            batch_limit = min(remaining, batch_limit)
        req = requests.get(endpoint, params={"limit": batch_limit, "offset": offset})
        req.raise_for_status()

        result = req.json()[key]
        if not result:
            break

        count = len(result)
        if remaining is not None:
            remaining -= count
        offset += count
        objects += result

    return objects


def get_document(document_id: int) -> Optional[dict]:
    doc_url = urllib.parse.urljoin(args.document_endpoint, str(doc_id))
    document_req = requests.get(doc_url)
    if document_req.status_code == 404:
        return
    document_req.raise_for_status()
    return document_req.json()


features = get_objects(args.feature_endpoint, batch_size=10)

annotated_docs = set()
for feature in features:
    true_docs = feature["documents"]["true"]
    false_docs = feature["documents"]["false"]
    annotated_docs.update(true_docs + false_docs)

annotated_docs = list(annotated_docs)
if args.limit:
    annotated_docs = annotated_docs[: args.limit]
for doc_id in tqdm.tqdm(annotated_docs):

    file_path = data_path / f"{doc_id}.pdf"
    if not file_path.exists():
        document = get_document(doc_id)
        if not document:
            warnings.warn(f"Could not request document with id '{doc_id}'")
            continue

        file_url = document["file_url"]
        if not file_url.startswith("http"):
            file_url = urllib.parse.urljoin(
                args.document_endpoint, document["file_url"]
            )
        file_id = document["id"]

        file_req = requests.get(file_url, stream=True, allow_redirects=False)
        file_req.raise_for_status()
        content_type = file_req.headers["content-type"]
        if "pdf" not in content_type:
            warnings.warn(f"Did not get a pdf when trying to download '{doc_id}'")
            continue
        with open(file_path, "wb") as f:
            for chunk in file_req.iter_content(chunk_size=4096):
                f.write(chunk)

    metadata = {}
    for feature in features:
        feature_name = feature["name"]
        if doc_id in feature["documents"]["true"]:
            metadata[feature_name] = True
        elif doc_id in feature["documents"]["false"]:
            metadata[feature_name] = False

    meta_path = data_path / f"{doc_id}.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
