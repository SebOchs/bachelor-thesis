import utils

TRAIN_PATH = 'out/train_loss_per_batch.npy'
VAL_PATH = 'out/val_loss_per_epoch.npy'
TRAIN_LOC = 'out/train.png'
VAL_LOC = 'out/val.png'

utils.plot(TRAIN_PATH, TRAIN_LOC, ['Train Loss'])
utils.plot(VAL_PATH, VAL_LOC, ['Macro F1','Weighted F1','Accuracy'])


"""
a = [x[0] for x in y3]
b = [x[1] for x in y3]
c = [x[2] for x in y3]

plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.savefig('val.png')
"""