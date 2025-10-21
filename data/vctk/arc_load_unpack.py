import os
import requests
from tqdm import tqdm
import zipfile
import tarfile

def download_file(url, output_path):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading")

    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("⚠️ Download size mismatch, file may be incomplete.")
    else:
        print(f"✅ Download complete: {output_path}")


def unpack_file(file_path, extract_to=None, delete_after=True):
    """Unpack .zip or .tar.gz files, optionally delete after unpacking."""
    # Default: extract to the same directory as the archive
    if extract_to is None:
        extract_to = os.path.dirname(os.path.abspath(file_path)) or "."
    else:
        os.makedirs(extract_to, exist_ok=True)
    
    print(f"📦 Unpacking {file_path} to {extract_to}")

    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Unpacking ZIP"):
                zip_ref.extract(member, extract_to)
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc="Unpacking TAR"):
                tar_ref.extract(member, extract_to)
    else:
        print("❌ Unsupported file format.")
        return

    print("✅ Unpacking complete.")

    if delete_after:
        try:
            os.remove(file_path)
            print(f"🧹 Deleted archive: {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}: {e}")


if __name__ == "__main__":
    # Example: replace with your own URL and output file name
    url = "http://www.udialogue.org/download/VCTK-Corpus.tar.gz"
    output_file = "VCTK-Corpus.tar.gz"

    download_file(url, output_file)
    unpack_file(output_file, delete_after=True)
