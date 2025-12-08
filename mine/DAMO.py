import sys
sys.path.append(".")
from lithobench.dataset import *
from lithobench.model import *

from lithobench.ilt.damoilt import DAMOILT

if __name__ == "__main__":
    """
    directly run this file to train the model
    """
    BATCH_SIZE = 4
    EPOCHS = 1
    IMAGE_SIZE = (1024, 1024)
    TRAIN_DATASET = "MetalSet"
    TEST_DATASET = "StdMetal"
    MODEL_NAME = "DAMOILT"

    train_loader, val_loader = loadersILT(TRAIN_DATASET, IMAGE_SIZE, batch_size=BATCH_SIZE, njobs=8)
    model = DAMOILT(size=IMAGE_SIZE)
    Folder = os.path.join("dev", f"{TRAIN_DATASET}_{MODEL_NAME}")

    # evaluate the randomly initialized model
    targets = evaluate.getTargets(samples=None, dataset=TRAIN_DATASET)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate the pretrained model
    model.pretrain(train_loader, val_loader, epochs=EPOCHS)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate the posttrained model
    model.train(train_loader, val_loader, epochs=EPOCHS)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate on StdMetal
    Folder = os.path.join("saved", f"{TEST_DATASET}_{MODEL_NAME}")
    targets = evaluate.getTargets(samples=None, dataset=TEST_DATASET)
    model.evaluate(targets, finetune=False, folder=Folder)