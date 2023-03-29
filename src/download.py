import json
import os

import requests
from tqdm import tqdm


# Define a function to download a file and retry if it fails
def download_file(url, filename):
    """
    Downloads a file from the given URL and saves it with the specified filename.

    If the file already exists, the function skips the download.

    Args:
        url (str): The URL of the file to download.
        filename (str): The name of the file to save.

    Returns:
        None"""

    if os.path.exists(filename):
        print(f"Skipping {filename}")
        return
    while True:
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=filename
            )
            with open(filename, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            print(f"Downloaded {filename}")
            break
        except Exception:
            print(f"Failed to download {filename}, retrying...")


# Define a function to download files from a directory
def download_files(directory):
    """
    Downloads multiple files from the given directory.

    The function creates the directory if it doesn't exist, loads the JSON data from a file within the directory, and sequentially downloads all the files specified in the JSON data.

    Args:
        directory (str): The directory to download the files into.

    Returns:
        None"""

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Load the JSON data from the file
    with open(f"{directory}/{directory}.json", encoding="utf-8") as f:
        json_data = json.load(f)
    # Download all the files sequentially
    for i, url in enumerate(json_data["files"], 1):
        filename = f"{directory}/{directory}_{i}.jsonl.gz"
        download_file(url, filename)


if __name__ == "__main__":
    # Download files from the "papers" directory
    download_files("papers")

    # Download files from the "citations" directory
    download_files("citations")
