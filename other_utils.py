
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# plots the training and validation loss and f1 score
def print_progress(epoch, i, tracking, num_total_batches):
    clear_output(wait=True)
    print('Epoch: ' + str(epoch))
    print('Batch: ' + str(i) + '/' + str(num_total_batches))

    loss, f1 = np.array(tracking['loss_train']), np.array(tracking['f1_train'])
    if len(loss) > 0:
        # smooth the loss and f1 with a moving average
        ma_size = loss.shape[0] // 5
        last_loss, last_f1 = loss[-1], f1[-1]
        loss = np.convolve(loss, np.ones((ma_size,))/ma_size, mode='valid')
        f1 = np.convolve(f1, np.ones((ma_size,))/ma_size, mode='valid')

        # plot the loss and f1 for the training
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 3, 1)
        plt.title('Training Loss: ' + format(last_loss, '.3f'))
        plt.plot(loss, label='Loss')
        plt.subplot(1, 3, 2)
        plt.title('Training Log Loss: ' + format(np.log(last_loss), '.3f'))
        plt.plot(np.log(loss), label='Log Loss')
        plt.subplot(1, 3, 3)
        plt.title('Training F1 Score: ' + format(last_f1, '.3f'))
        plt.plot(f1, label='F1 Score')
        plt.show()

    # plot the loss and f1 for the validation
    loss, f1 = np.array(tracking['loss_eval']), np.array(tracking['f1_eval'])
    if len(loss) > 0:
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 3, 1)
        plt.title('Evaluation Loss: ' + format(loss[-1], '.3f'))
        plt.plot(loss, label='Loss')
        plt.subplot(1, 3, 2)
        plt.title('Evaluation Log Loss: ' + format(np.log(loss[-1]), '.3f'))
        plt.plot(np.log(loss), label='Log Loss')
        plt.subplot(1, 3, 3)
        plt.title('Evaluation F1 Score: ' + format(f1[-1], '.3f'))
        plt.plot(f1, label='F1 Score')
        plt.show()