from typing import List

from llama_index.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core import Settings


class InputReader:
    def __init__(self, input_dir: str) -> None:
        self.reader = SimpleDirectoryReader(input_dir=input_dir)

    def parse_documents(
        self,
        show_progress: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Document]:
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        documents = self.reader.load_data(show_progress=show_progress)
        return documents
