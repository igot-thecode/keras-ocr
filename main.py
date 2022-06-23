import matplotlib.pyplot as plt
import keras_ocr
import itertools



# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images, needs to be iterable
images = [
    keras_ocr.tools.read(url) for url in [
        'images/i3.png', 'images/i4.png'
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)
total_findings = list(itertools.chain(*prediction_groups))

print(total_findings)
all_words = []
for pred in total_findings:
        print(pred[0])
        all_words.append(pred[0])

print(''.join(all_words).upper())
# Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

# plt.show()