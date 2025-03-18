import requests

data_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
def download_dataset():
    response = requests.get(data_url)
    if response.status_code == 200:
        with open("data.zip", "wb") as f:
            f.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Failed to download dataset.")

def unzip_dataset(target_dir):
    import zipfile
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Dataset unzipped successfully.")

download_dataset()
unzip_dataset("data/")