import torch
import torch.nn as nn
from model import Model
import torch.optim as optim
from model_utils import seed_torch, reaData
from torch.utils.data import DataLoader
import warnings
import datetime
warnings.filterwarnings('ignore')
seed_torch(0)

with open('output/info.txt', 'w') as source:
    source.write('Training:'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+
                 '\n===================================================\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# loading data
train = DataLoader(dataset=reaData('zhihu', 'train'), batch_size=8, shuffle=True)
# test = DataLoader(dataset=reaData('zhihu', 'test'), batch_size=1, shuffle=True)
hidden_units = [512, 64, 2]
mb_units = [2048, 512, 100]
# loading model
model = Model(qs_input_dim=320, as_input_dim=142, mb_dim=4980, hidden_units=hidden_units,
              mb_units=mb_units, device=device).float()
# qsMDL = TransMdl(n_layers=6, n_head=2, seq_len=9, d_model=142, d_inner=512, fc1=142,
#                 d_out=2, dropout=0.1, d_ivt=320, d_mb=4980)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 100

for epoch in range(epochs):

    running_loss = 0
    i = 0
    for data, label in train:
        qes, ans, ivt, meb, target = data[0].to(device), data[1].to(device), data[2].to(device),\
                                     data[3].to(device), label.to(device)

        optimizer.zero_grad()

        out = model(qes.float(), ans.float(), ivt.float(), meb.float())
        loss = criterion(out, target.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        i += 1
        if i % 100 == 99:
            line = 'Epoch:%d Iter:%d loss:%.5f Total:%.5f' % (epoch + 1, i + 1, running_loss / 100, running_loss)
            print(line)
            with open('output/info.txt', 'a') as source:
                source.write(line+'\n')
            # print(torch.sigmoid(out))
            # print(label)
            running_loss = 0
        if i % 1100 == 1099:
            print('-----Save model-----')
            torch.save(model, 'output/my_model.ptm')

    print('Finish.')