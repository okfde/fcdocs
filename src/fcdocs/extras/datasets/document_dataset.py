import json
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, NamedTuple

from filingcabinet.pdf_utils import PDFProcessor
from kedro.io import AbstractDataSet
from PIL import Image

PAGE_FEED_MARKER = "\n\x12\n"
THUMBNAIL_WIDTH = 180

logger = logging.getLogger(__name__)


def get_pdf_processor(pdf_filepath) -> PDFProcessor:
    config = {}
    return PDFProcessor(str(pdf_filepath), language="de", config=config)


class DocumentData(NamedTuple):
    """
    PDFPage to store read PDF data
    """

    text: str
    image: Image.Image
    meta: Dict[str, str]


class DocumentDataSet(AbstractDataSet):
    """``DocumentDataset`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> DocumentDataset(filepath='/img/file/path.pdf')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of DocumentDataset to load / save PDF data at the given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        self._filepath = PurePosixPath(filepath)

    def _load(self) -> DocumentData:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array.
        """
        text_filepath, image_filepath, meta_filepath = self.get_filepaths()
        text_exists = text_filepath.exists()
        image_exists = image_filepath.exists()
        meta_exists = meta_filepath.exists()
        existing_meta = {}
        if meta_exists:
            with open(meta_filepath) as f:
                existing_meta = json.load(f)
        if not text_exists or not image_exists:
            logger.info("Processing PDF %s", self._filepath)

            doc_data = self._load_pdf()
            if meta_exists:
                existing_meta.update(doc_data.meta)
                doc_data = DocumentData(
                    text=doc_data.text, image=doc_data.image, meta=existing_meta
                )
            # Save generated data for next load
            self._save(doc_data)
            return doc_data

        with open(text_filepath) as f:
            text = f.read()

        if meta_exists:
            with open(meta_filepath) as f:
                meta = json.load(f)
        else:
            meta = {}

        img = Image.open(image_filepath)
        img_in_mem = img.copy()
        img.close()

        return DocumentData(text=text, image=img_in_mem, meta=meta)

    def get_filepaths(self):
        return (
            Path(self._filepath.with_suffix(".txt")),
            Path(self._filepath.with_suffix(".png")),
            Path(self._filepath.with_suffix(".json")),
        )

    def _load_pdf(self):
        pdf = get_pdf_processor(self._filepath)
        meta = pdf.get_meta()
        meta["_num_pages"] = pdf.num_pages
        meta["size"] = os.path.getsize(self._filepath)
        for _page_number, wand_image in pdf.get_images(pages=[1]):
            wand_image.transform(resize="{}x".format(THUMBNAIL_WIDTH))
            img_blob = wand_image.make_blob("RGB")
            image = Image.frombytes("RGB", wand_image.size, img_blob)
        page_texts = PAGE_FEED_MARKER.join(pdf.get_text())
        return DocumentData(text=page_texts, image=image, meta=meta)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _save(self, data: DocumentData) -> None:
        """Saves image data to the specified filepath"""
        text_filepath, image_filepath, meta_filepath = self.get_filepaths()
        with open(text_filepath, "w") as f:
            f.write(data.text)
        data.image.save(image_filepath)
        with open(meta_filepath, "w") as f:
            json.dump(data.meta, f)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(filepath=self._filepath)
