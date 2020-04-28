import os
import glob

images = glob.glob("data/Images/**/*.jpg", recursive=True)
classes, train, test = dict(), dict(), dict()
n = 5

# Split images into classes based on path
for image in images:
    path, name = os.path.split(image)
    if path not in classes:
        classes[path] = []
    classes[path].append(name)

# Use first n images of each class as test and rest as train
for c in classes:
    test[c] = classes[c][:n]
    train[c] = classes[c][n:]

# Write to files
for t in ['train', 'test']:
    with open(f'data/{t}.txt', 'w') as f:
        d = eval(t)
        for c in d:
            for name in d[c]:
                print(os.path.join(c, name), file=f)
