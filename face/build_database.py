import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

OWNER_FOLDER = "face_db/owner"
OUTPUT_FILE = "embeddings/owner_embedding.npy"

os.makedirs("embeddings", exist_ok=True)


def main():

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []

    image_files = sorted(os.listdir(OWNER_FOLDER))

    for image_name in tqdm(image_files):

        path = os.path.join(OWNER_FOLDER, image_name)

        image = cv2.imread(path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) != 1:
            continue

        embeddings.append(faces[0].embedding)

    if len(embeddings) == 0:
        raise Exception("No valid faces found.")

    embeddings = np.array(embeddings)

    owner_embedding = np.mean(embeddings, axis=0)

    owner_embedding = owner_embedding / np.linalg.norm(owner_embedding)

    np.save(OUTPUT_FILE, owner_embedding)

    print()
    print("=" * 50)
    print(f"Images Processed : {len(image_files)}")
    print(f"Valid Faces      : {len(embeddings)}")
    print("Database Created Successfully")
    print("=" * 50)


if __name__ == "__main__":
    main()