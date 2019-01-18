import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import sys

from utils import get_device, format, plot
from models import get_model
from dataset import Dataset

MODEL_TYPE = 'NEG' # CBOW | NGRAM | SKIPGRAM | NEG
CONTEXT_SIZE = 2
EMBEDDING_DIM = 300
LR = 0.001
BATCH_SIZE = 2
NUM_EPOCHS = 100

# SEED
torch.manual_seed(1)


# SETUP
dataset = Dataset(MODEL_TYPE, CONTEXT_SIZE)
dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **dataloader_kwargs)
criterion = CrossEntropyLoss() if MODEL_TYPE != 'NEG' else BCEWithLogitsLoss()
model = get_model(dataset.vocab_size, MODEL_TYPE, CONTEXT_SIZE, EMBEDDING_DIM)
optimizer = Adam(model.parameters(), lr=LR)
# print(sum(p.numel() for p in model.parameters())) # NUM PARAMS
device = get_device()
epoch = 0


# TRAIN
for epoch in range(epoch + 1, NUM_EPOCHS + 1):
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        data, target = batch
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        if MODEL_TYPE == 'NEG':
            output = output.squeeze(1)

        target = format(target, MODEL_TYPE)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            msg = (
                f"Train Epoch: {epoch} "
                f"[{batch_idx * BATCH_SIZE}/{len(dataloader.dataset)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
            print(msg, file=sys.stderr)

    print(f'====> Train set loss: {total_loss / len(dataloader):.4f}', file=sys.stderr)

# eval
with torch.no_grad():
    model.eval()
    data, target = dataset.samples, dataset.targets
    data = data.to(device)

    output = model(data)
    target = format(target, MODEL_TYPE, eval=True)

    if MODEL_TYPE != 'NEG':
        _, pred = output.topk(1)
    else:
        pred = (output >= 0.5).float()

    correct = (pred==target).sum().item()
    print(f'ACC: {correct} / {target.size(0)} [{100*correct / target.size(0):.2f}%]')

    if EMBEDDING_DIM == 2:
        plot(model, dataset.vocab)





# EOF
