import imageio
import numpy as np
import torchvision

import util
import visual_words
import visual_recog
import deep_recog

def random_image():
    train_data = np.load('../data/train_data.npz')
    path = '../data/' + train_data['image_names'][np.random.choice(1440)][0]
    print(path)
    return path

if __name__ == '__main__':

    num_cores = util.get_num_CPU()

    # path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    # path_img = random_image()
    # path_img = "../data/auditorium/sun_afcwsapxpihedtku.jpg" # 4 channels
    # image = imageio.imread(path_img)
    # image = image.astype('float')/255
    # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)

    # visual_words.compute_dictionary(num_workers=num_cores)
    
    # dictionary = np.load('dictionary.npy')
    # wordmap = visual_words.get_visual_words(image, dictionary)
    # wordmap = np.load('wordmap.npy')
    # util.save_wordmap(wordmap, '../writeup/' + path_img.split('/')[-1])

    # UNIT TESTING
    # visual_recog.get_feature_from_wordmap(wordmap, dictionary.shape[0])
    # visual_recog.get_feature_from_wordmap_SPM(wordmap, 3, dictionary.shape[0])
    # visual_recog.get_image_feature(path_img, dictionary, 3, dictionary.shape[0])

    # visual_recog.build_recognition_system(num_workers=num_cores)
    # conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    # deep_recog.build_recognition_system(vgg16, num_workers=num_cores//2)
    # conf, accuracy = deep_recog.evaluate_recognition_system(vgg16, num_workers=num_cores//2)
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())

    deep_recog.evaluate_custom_implementation(vgg16)


