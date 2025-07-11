
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


MODELS = [#'NLI_FullLinear_16M', 'NLI_PairsBasic_16M', 'NLI_Heads_16M', 'NLI_MiniHeads_16M',
            'NLI_FullLinear_4M', 'NLI_PairsBasic_4M', 'NLI_Heads_4M', 'NLI_MiniHeads_4M',
            'NLI_FullLinear_300k', 'NLI_PairsBasic_300k', 'NLI_Heads_300k', 'NLI_MiniHeads_300k']

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
    nli_f1 = np.array(tracking['nli_f1'])
    nli_accuracy = np.array(tracking['nli_accuracy'])
    average_enough_retrieved = np.array(tracking['average_enough_retrieved'])
    average_total_retrieved = np.array(tracking['average_total_retrieved'])
    loss2 = np.array(tracking['average_loss2'])

    # smooth everything with a moving average
    if ma_ratio is not None:
        ma_size = int(average_enough_retrieved.shape[0] * ma_ratio)
        if ma_size > 0:
            new_nli_f1 = np.zeros((nli_f1.shape[0] - ma_size + 1, len(MODELS)))
            new_nli_accuracy = np.zeros((nli_accuracy.shape[0] - ma_size + 1, len(MODELS)))
            new_loss2 = np.zeros((loss2.shape[0] - ma_size + 1, len(MODELS)))
            for i in range(len(MODELS)):
                new_nli_f1[:, i] = np.convolve(nli_f1[:, i], np.ones((ma_size,))/ma_size, mode='valid')
                new_nli_accuracy[:, i] = np.convolve(nli_accuracy[:, i], np.ones((ma_size,))/ma_size, mode='valid')
                new_loss2[:, i] = np.convolve(loss2[:, i], np.ones((ma_size,))/ma_size, mode='valid')
            nli_f1 = new_nli_f1
            nli_accuracy = new_nli_accuracy
            loss2 = new_loss2
            average_enough_retrieved = np.convolve(average_enough_retrieved, np.ones((ma_size,))/ma_size, mode='valid')
            average_total_retrieved = np.convolve(average_total_retrieved, np.ones((ma_size,))/ma_size, mode='valid')
            


    # plot the metrics
    plt.figure(figsize=(10, 5))
    plt.title('BCELoss for ' + set_name + ' Set')
    for i, model in enumerate(MODELS):
        plt.plot(loss2[:, i], label=model)
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.title('F1 Score for ' + set_name + ' Set')
    for i, model in enumerate(MODELS):
        plt.plot(nli_f1[1:, i], label=model)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Accuracy for ' + set_name + ' Set')
    for i, model in enumerate(MODELS):
        plt.plot(nli_accuracy[1:, i], label=model)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Retrieval Metrics for ' + set_name + ' Set')
    plt.plot(average_enough_retrieved, label='Average Enough Retrieved')
    plt.plot(average_total_retrieved, label='Average Total Retrieved')
    plt.legend()
    plt.show()




    