import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(
        self,
        embedding_path="embeddings/owner_embedding.npy",
        threshold=0.60,
    ):

        self.threshold = threshold

        self.owner_embedding = np.load(embedding_path)
        self.owner_embedding = (
            self.owner_embedding /
            np.linalg.norm(self.owner_embedding)
        )

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def recognize(self, crop):

        if crop is None or crop.size == 0:
            return "No Face", 0.0, None

        faces = self.app.get(crop)

        if len(faces) == 0:
            return "No Face", 0.0, None

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        similarity = float(
            np.dot(
                embedding,
                self.owner_embedding
            )
        )

        x1, y1, x2, y2 = map(int, face.bbox)

        face_bbox = (x1, y1, x2, y2)

        if similarity >= self.threshold:
            return "Owner", similarity, face_bbox

        return "Unknown", similarity, face_bbox