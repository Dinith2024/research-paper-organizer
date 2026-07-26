# download_onnx_model.py
from pathlib import Path
import httpx, shutil

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

emb = ONNXMiniLM_L6_V2()
download_dir = Path("data/chroma/.embedding_cache") / emb.MODEL_NAME
download_dir.mkdir(parents=True, exist_ok=True)
fname = download_dir / emb.ARCHIVE_FILENAME

url = emb.MODEL_DOWNLOAD_URL
httpx.embedding_function._client_config = {"timeout": httpx.Timeout(300.0)}
with httpx.stream("GET", url, timeout=300.0) as r:
    r.raise_for_status()
    with open(fname, "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)

shutil.unpack_archive(str(fname), extract_dir=str(download_dir))
print("Model downloaded and extracted to", download_dir)