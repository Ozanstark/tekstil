import os
import urllib.request
import tarfile

URLS = {
    'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951468/carpet.tar.xz',
    'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951508/leather.tar.xz',
    'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951498/grid.tar.xz'
}

DATA_DIR = "/Users/oes/tekstil/data/mvtec"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for name, url in URLS.items():
        tar_path = os.path.join(DATA_DIR, f"{name}.tar.xz")
        out_dir = os.path.join(DATA_DIR, name)
        
        if os.path.exists(out_dir):
            print(f"{name} is already extracted. Skipping.")
            continue
            
        if not os.path.exists(tar_path):
            print(f"Downloading {name} from MVTec (may take a few minutes)...")
            try:
                urllib.request.urlretrieve(url, tar_path)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
                
        print(f"Extracting {name}...")
        try:
            with tarfile.open(tar_path, "r:xz") as tar:
                tar.extractall(path=DATA_DIR)
            print(f"Successfully extracted {name}.")
        except Exception as e:
            print(f"Extraction failed for {name}: {e}")

if __name__ == "__main__":
    download_and_extract()
    print("All downloads completed.")
