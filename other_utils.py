
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# plots the training and validation loss and f1 score
def print_progress(epoch, batch, num_total_batches, tracking_train=None, tracking_eval=None, ma_ratio=None):
    clear_output(wait=True)
    print('Epoch: ' + str(epoch))
    print('Batch: ' + str(batch) + '/' + str(num_total_batches))

    if tracking_train is not None:
        print_tracker(tracking_train, 'Training', ma_ratio=ma_ratio)
    if tracking_eval is not None:
        print_tracker(tracking_eval, 'Validation', ma_ratio=ma_ratio)
        

def print_tracker(tracking, set_name, ma_ratio=None):
    nli_accuracy = np.array(tracking['nli_accuracy'])
    nli_f1 = np.array(tracking['nli_f1'])
    overall_accuracy = np.array(tracking['overall_accuracy'])
    overall_f1 = np.array(tracking['overall_f1'])
    average_enough_retrieved = np.array(tracking['average_enough_retrieved'])
    average_total_retrieved = np.array(tracking['average_total_retrieved'])
    loss1 = np.array(tracking['average_loss1'])
    loss2 = np.array(tracking['average_loss2'])

    # smooth everything with a moving average
    if ma_ratio is not None:
        ma_size = int(nli_accuracy.shape[0] * ma_ratio)
        if ma_size > 0:
            nli_accuracy = np.convolve(nli_accuracy, np.ones((ma_size,))/ma_size, mode='valid')
            nli_f1 = np.convolve(nli_f1, np.ones((ma_size,))/ma_size, mode='valid')
            overall_accuracy = np.convolve(overall_accuracy, np.ones((ma_size,))/ma_size, mode='valid')
            overall_f1 = np.convolve(overall_f1, np.ones((ma_size,))/ma_size, mode='valid')
            average_enough_retrieved = np.convolve(average_enough_retrieved, np.ones((ma_size,))/ma_size, mode='valid')
            average_total_retrieved = np.convolve(average_total_retrieved, np.ones((ma_size,))/ma_size, mode='valid')
            loss1 = np.convolve(loss1, np.ones((ma_size,))/ma_size, mode='valid')
            loss2 = np.convolve(loss2, np.ones((ma_size,))/ma_size, mode='valid')

    # plot the metrics
    plt.figure(figsize=(10, 5))
    plt.title('Loss for ' + set_name + ' Set')
    plt.plot(loss1, label='Loss 1')
    plt.plot(loss2, label='Loss 2')
    plt.plot(loss1 + loss2, label='Total Loss')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.title('Retrieval Metrics for ' + set_name + ' Set')
    plt.plot(average_enough_retrieved, label='Average Enough Retrieved')
    plt.plot(average_total_retrieved, label='Average Total Retrieved')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.title('Accuracy and F1 Score for ' + set_name + ' Set')
    plt.plot(nli_accuracy, label='NLI Accuracy')
    plt.plot(nli_f1, label='NLI F1 Score')
    plt.plot(overall_accuracy, label='Overall Accuracy')
    plt.plot(overall_f1, label='Overall F1 Score')
    plt.legend()
    plt.show()





    