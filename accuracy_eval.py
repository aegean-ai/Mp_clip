# read the results
import glob
import os
import logging

logger = logging.getLogger()
logger.setLevel('INFO')

# logging.basicConfig(level=logging.INFO)
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')


from natsort import natsorted
from skimage import io

# from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
# from sklearn.metrics import *
import numpy as np

label_folder = r'L:\NewYorkCity_sidewalks\sidewalks\Test'
label_suf = 'TIF'
label_images = glob.glob(os.path.join(label_folder, '*.' + label_suf))
results = 'png'
results_folder = r'L:\NewYorkCity_sidewalks\COCO\Test256\classified_padding10_432\merge\binary'
results_images = glob.glob(os.path.join(results_folder, '*' + '.png'))
# print(os.path.join(results_folder, '\*.png'))
results_images = natsorted(results_images)
label_images = natsorted(label_images)
print('result_images: ', results_images[:3])
print('result_images: ', label_images[:3])

# with open(r'H:\temp', 'w') as f:
#     f.writelines('khh')

f = open(r'K:\OneDrive_NJIT\OneDrive - NJIT\Ipynb\test.txt', 'a')
f.close()

def metrics(predictions, gts, label_values, report_path):
    print('range(len(gts))')
    print(range(len(gts)))

    print('range(len(predictions))')
    print(range(len(predictions)))

    print('range(len(label_values))')
    print(range(len(label_values)))

    # report = open(MAIN_FOLDER + 'Test_all_report_100tiles_QS.txt', 'w')
    report = open(report_path, 'w')
    report.writelines('Train ids: ' + str("Confusion matrix :\n"))

    cm = confusion_matrix(gts, predictions, label_values)

    f1 = f1_score(gts, predictions, average='micro')
    print("F1_score:  micro")
    print(f1)

    f1 = f1_score(gts, predictions, average='macro')
    print("F1_score:  macro")
    print(f1)

    f1 = f1_score(gts, predictions, average=None)
    print("F1_score: None")
    print(f1)

    #     accur = accuracy_score(gts, predictions)
    #     print("accuracy_score:  ")
    #     print(accur)

    #     rpt = classification_report(gts, predictions, LABELS)
    #     print("classification_report:  ")
    #     print(accur)

    print("Confusion matrix :")
    report.writelines(str("Confusion matrix : \n"))
    print(cm)
    report.writelines(str(cm) + '\n')
    report.writelines('----------- \n ')

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    report.writelines("{} pixels processed \n".format(total))
    report.writelines("Total accuracy : {}% \n".format(accuracy))
    report.writelines("---\n")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    class_accuracy = np.zeros(len(label_values))
    class_recall = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
            class_accuracy[i] = cm[i, i] / np.sum(cm[:, i])
            class_recall[i] = cm[i, i] / np.sum(cm[i, :])
        except:
            # Ignore exception if there is no element in class i for test set
            pass

    report.writelines("F1Score : \n")
    print("F1Score :")

    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(LABELS[label_values[l_id]], score))
        report.writelines("{}: {} \n".format(LABELS[label_values[l_id]], score))
    print("---")

    print("class_accuracy :")
    report.writelines("\n class_accuracy : \n")
    for l_id, score in enumerate(class_accuracy):
        print("{}: {}".format(LABELS[label_values[l_id]], score))
        report.writelines("{}: {} \n".format(LABELS[label_values[l_id]], score))
    print("---")

    print("class_recall :")
    report.writelines("\n class_recall : \n")
    for l_id, score in enumerate(class_recall):
        print("{}: {}".format(LABELS[label_values[l_id]], score))
        report.writelines("{}: {} \n".format(LABELS[label_values[l_id]], score))
    print("---")

    print("\nClass summary :")
    report.writelines("\nClass summary :\n")
    for i in range(len(label_values)):
        print('Correct, Ground truth, Predict: ', LABELS[label_values[i]], cm[i, i], np.sum(cm[i, :]), np.sum(cm[:, i]))
        report.writelines('Correct, Ground truth, Predict: {}: {}, {}, {}\n'.format(LABELS[label_values[i]], cm[i, i],
                                                                                    np.sum(cm[i, :]), np.sum(cm[:, i])))

    report.writelines("---\n")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))

    report.writelines("Kappa: " + str(kappa) + '\n')

    report.close()

    return accuracy


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1


def calcu_results_metric(label_imgs, result_imgs):
    if not isinstance(label_imgs, list):
        label_imgs = [label_imgs]
    if not isinstance(result_imgs, list):
        result_imgs = [result_imgs]

    label_names = []
    for file in label_imgs:
        label_names.append(os.path.splitext(os.path.basename(file))[0])

    print('label_names, len()： ', len(label_names), label_names)

    for file in result_imgs:
        base_name = os.path.splitext(os.path.basename(file))[0]
        idx_lab = find_element_in_list(base_name, label_names)
        if idx_lab > -1:
            #            pass
            print('base_name: ', base_name)

            res = io.imread(file).flatten()
            lab = io.imread(label_imgs[idx_lab]).flatten()
            lab = lab.astype(np.uint8)

            print("len(res), len(lab): ", len(res), len(lab))

            print("Label unique values: ", np.unique(lab))
            print("Result unique values: ", np.unique(res))

            metrics(res, lab, ['background', 'sidewalk'],
                    r'H:\temp')


#             print("classification_report: \n", metrics.classification_report(lab, res))

#             accur = accuracy_score(lab, res)
#             print('accur:', accur)

#             recall = metrics.recall_score(lab, res, pos_label=1)
#             print('recall:', recall)


#             precision = precision_score(lab, res, pos_label=1)
#             print('precision:', precision)

#             cm = confusion_matrix(gts, predictions, label_values)

#             f1 = f1_score(gts, predictions, average='micro')
#             print("F1_score:  micro")
#             print(f1)

#             f1 = f1_score(lab, res, average='macro')
#             print("F1_score:  macro")
#             print(f1)

#             f1 = f1_score(gts, predictions, average=None)
#             print("F1_score: None")
#             print(f1)

#            logger.info(f'base_name: {base_name}')
#             res = io.imread(file)
#             lab = io.imread(label_imgs[idx_lab])


calcu_results_metric(label_images, results_images)