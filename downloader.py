import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from config import Config

def download_images(num_images=1000):
    save_dir = Config.RAW_DATA_PATH
    os.makedirs(save_dir, exist_ok=True)

    meta_dir = os.path.dirname(Config.METADATA_CSV)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)

    print(f"Downloading {num_images} images from WikiArt to {save_dir}...")

    # Load dataset (streaming to avoid loading full dataset into memory)
    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    except Exception as e:
        print("Failed to connect to HuggingFace. Check your network or proxy.")
        print("Error:", e)
        return

    rows = []
    count = 0

    for item in tqdm(dataset):
        if count >= num_images:
            break

        try:
            image = item["image"]

            # Convert to RGB and skip images that are too small
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size[0] < 256 or image.size[1] < 256:
                continue

            # Filter: remove Unknown Artist / Unknown Genre
            artist = int(item.get("artist", -1))
            genre_id = int(item.get("genre", -1))
            style_label = str(item.get("style", -1))

            if artist == 0:
                continue

            if genre_id == 10:
                continue

            filename = f"{style_label}_{count}.jpg"
            save_path = os.path.join(save_dir, filename)

            image.save(save_path, quality=95)

            rows.append({
                "image_id": count,
                "file_path": save_path,
                "artist": artist,
                "genre": genre_id,      
                "style": style_label,   
            })

            count += 1

        except Exception as e:
            print(f"Error saving image: {e}")
            continue

    print(f"Successfully downloaded {count} images.")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Config.METADATA_CSV, index=False)
        print(f"Saved metadata to {Config.METADATA_CSV}")
    else:
        print("No images were saved, metadata.csv not created.")


if __name__ == "__main__":
    download_images(num_images=20000)