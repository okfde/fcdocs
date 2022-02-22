#!/bin/bash

BASE_URL=https://fragdenstaat.de/api/v1/attachment/?filetype=application/pdf

DATA_PATH=data/01_raw/pdfdocs_01

curl "$BASE_URL" | jq -c -r '.objects[]' | while read -r att; do
    fileid="$(echo $att | jq '.id')"
    echo $fileid
    echo $att > "$DATA_PATH/${fileid}.json"
    url="$(echo $att | jq -r '.file_url')"
    wget -O "$DATA_PATH/${fileid}.pdf" --no-clobber $url
done
