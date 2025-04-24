import os
import gdown
import subprocess

FILE_ID = "1e_W2qbtrmV3bxohJhJcqYQJzeGngRXG0"
OUTPUT_DIR = "assets/data/"
ZIP_NAME = "tetweave_evaluation_dataset.zip"

def download_with_gdown(file_id, output):
    """Download using gdown with progress"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

def extract_zip(zip_path, output_dir):
    """Extract using unzip"""
    print(f"Extracting to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", zip_path, "-d", output_dir], check=True)

def main():
    download_with_gdown(FILE_ID, ZIP_NAME)
    extract_zip(ZIP_NAME, OUTPUT_DIR)
    os.remove(ZIP_NAME)
    print(f"\nDone! Dataset available in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()