import pathlib
import matplotlib.pyplot as plt
import keras_ocr
import itertools
import concurrent.futures
import multiprocessing


# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


def get_image_data(image_url):
    #print(image_url)
    # Get a set of three example images, needs to be iterable
    image = [keras_ocr.tools.read(image_url)] #syntax used to be able to work with just one image at a time

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(image)
    #print(prediction_groups)
    total_findings = list(itertools.chain(*prediction_groups)) #flatten the data

    #print(total_findings)
    all_words = []
    for pred in total_findings:
            #print(pred[0])
            all_words.append(pred[0])
    reading = ''.join(all_words).upper().replace(' ','').replace('O','0')

    return (image_url, reading)
# Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

# plt.show()
def main():
    image_urls = []
    results = []
    images_info = []
    for path in pathlib.Path('images').iterdir():
        image_urls.append(str(path))

    with open('license_tests.csv','r') as test_file:
        for line in test_file:
            test_tuple = tuple(line.strip().split(','))
            images_info.append(test_tuple)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()//4) as exec:
        results = list(exec.map(get_image_data, image_urls))
        #print(results)
    #print(results)
    #print(images_info)
    correct = 0
    total = len(images_info)
    for url, prediction in results:
        folder, file = url.split('/')
        for image_info in images_info:
            if image_info[0] == file:
                if image_info[1] in prediction:
                    correct += 1
    print(f"{(correct/total)* 100:.3f}% correct")
    

if __name__ == '__main__':
    main()