from insightface.app import FaceAnalysis

app = FaceAnalysis(name = 'buffalo_l')
app.prepare(ctx_id = -1)
print('Model loaded successfully')

import cv2

img = cv2.imread('office.jpeg')
faces = app.get(img)

print(f'Detected {len(faces)} face(s)')

for face in faces:
    embedding = face.embedding
    print('Embedding shape:', embedding.shape)
    print('First 5 values:', embedding[:5])
