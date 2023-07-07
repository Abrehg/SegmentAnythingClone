import numpy as np

vert = 5
horiz = 9

encodings = np.random.randint(0,20,(vert, horiz, 1024))
print(np.shape(encodings))


mask = np.random.rand(vert, horiz,1)
print(np.shape(mask))
mask = mask > 0.75
print(encodings)
encodings = np.multiply(encodings, mask)
print(encodings)