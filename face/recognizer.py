import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        self.owner_embedding = np.load(
            "embeddings/owner_embedding.npy"
        )

        self.owner_embedding = (
            self.owner_embedding
            / np.linalg.norm(self.owner_embedding)
        )

        self.threshold = 0.60

    def recognize(self, image):

        faces = self.app.get(image)

        if len(faces) == 0:
            return None, 0.0

        face = max(
            faces,
            key=lambda x: (
                x.bbox[2] - x.bbox[0]
            ) * (
                x.bbox[3] - x.bbox[1]
            )
        )

        embedding = face.embedding

        embedding = embedding / np.linalg.norm(embedding)

        similarity = np.dot(
            embedding,
            self.owner_embedding
        )

        if similarity >= self.threshold:

            return "Owner", float(similarity)

        return "Unknown", float(similarity)