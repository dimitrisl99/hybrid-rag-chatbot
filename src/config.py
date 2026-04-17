from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index_numpy"

COLLECTION_NAME = "rag_papers"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
NORMALIZE = True
DEVICE = "cpu"
BATCH_SIZE = 16

PDF_TARGET_CHARS = 1200
PDF_MAX_CHARS = 1800
PDF_MIN_CHARS = 500
PDF_SIM_THRESHOLD = 0.78
PDF_WINDOW_SENTENCES = 3

MIN_CHUNK_LENGTH = 80
RESET_COLLECTION = True

USE_FAISS = True
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"