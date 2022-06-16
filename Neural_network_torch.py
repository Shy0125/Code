import torch, numpy as np, matplotlib.pyplot as plt, time, pandas as pd, pygmt
from torch.utils import data as tdata
from torch import nn
from copy import deepcopy
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error as Mse

def try_all_gpus():
    '''Return all available GPUs, or [mx.cpu()] if there is no GPU.'''
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
def read(filename,num_clo):
    '''
    File read function.
    :param filename: File path
    :param num_clo: File cloumn numbers
    :return: 2D array
    '''
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        lines = [float(ii) for i in lines for ii in i]
        lines = np.array(lines).reshape(-1,num_clo)
    return lines

def log_rmse(net, features, labels):
    '''
    Return loss for printing.
    :param net: Neural network model
    :param features: Feature data
    :param labels: Label data
    :return: Loss
    '''
    loss = nn.MSELoss()
    clipped_preds = torch.clamp(net(features.to(device[0])).abs(), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels.abs().to(device[0]))))
    return rmse.item()

def train_regression(net, train_iter, test_iter, lr, device, num_epochs):
    '''Train and evaluate a model.'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 80, eta_min= lr*0.1)
    loss = nn.MSELoss()

    Time, num_batches = [], len(train_iter)
    Train_l, Test_l, X_length, Lr= [],[],[],[]
    for epoch in range(num_epochs):
        net.train()
        start = time.time()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            X_length.append(X.shape[0])
        scheduler.step()
        Train_l.append(log_rmse(net,train_features.data,train_labels.data))
        Test_l.append(log_rmse(net,test_features,test_labels))
        Time.append(time.time() - start)
        Lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        print('epoch %d, train loss %.4F, test loss %.4f, time %.1f'%(epoch+1, Train_l[epoch], Test_l[epoch], Time[epoch]))
    print(f'{sum(X_length) / sum(Time):.1f} examples/sec 'f'on {str(device)}')
    return Train_l, Test_l, Lr

def predict(net, X):
    '''Predicion function based on trained net'''
    net.eval()
    Y = []
    for x in np.array_split(X,10):
        x = fun.transform(x)
        y = net(torch.Tensor(x).to(device[0]))
        Y.append(y.detach().cpu().numpy())
    return np.concatenate([*Y],0)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(2,16),
            nn.Linear(16,256),)
        self.block2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.Linear(16, 256), )
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, input, *args):
        temp = torch.cat([self.block1(input[:,[0,2]]), self.block2(input[:,[1,3]]) ], dim=1)
        return self.block3(temp)

def config():
    '''Return parameter for training.'''
    ctx, lr, wd, batch_size, epoch_num = try_all_gpus(), 0.005, 0, 512, 100
    net = model()
    return net, ctx, lr, wd, batch_size, epoch_num

#======================Data read====================================
Data = read(r"Train_dataset.txt", 7)
ind = read(r"Ind.txt", 1)

ind = ind.astype(int).tolist()
Ind = []
for i in ind:
    Ind += i

size ,I, R = [361,481], '1m', '140/148/10/16'
#=======================Data process====================================
df = pd.DataFrame(Data)

fun1 = MinMaxScaler()
label = deepcopy(df[2])

fun = StandardScaler()
features = df.drop(columns=[0,1,2])
features = fun.fit_transform(features)
features = pd.DataFrame(features)

test_features = torch.tensor(features.iloc[Ind,:].values).to(torch.float)
test_labels = torch.tensor(label.iloc[Ind].values).to(torch.float).reshape(-1,1)#reshape很重要，不然触发broadcast机制。
features.drop(Ind, inplace = True)
label.drop(Ind, inplace = True)
train_features = torch.tensor(features.values).to(torch.float)#将类型设置为float是因为神经网络的权重是float
train_labels = torch.tensor(label.values).to(torch.float).reshape(-1,1)

train_data = tdata.TensorDataset(train_features,train_labels)
test_data = tdata.TensorDataset(test_features,test_labels)
#=================================Model=======================

net,device,lr,wd,batch_size,epoch_num=config()

train_iter = tdata.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_iter = tdata.DataLoader(test_data,batch_size=batch_size,)

#================================Training===========================
a,b,c = train_regression(net, train_iter, test_iter, lr=lr, device=device[0], num_epochs=epoch_num)

#================================Prediction==========================

Pred = read(r"Prediction_dataset.txt", 7)
X = Pred[:,[3,4,5,6]]
Y = predict(net, X)
Y=Y.reshape(size)
plt.figure()
plt.imshow(Y)
plt.jet()
plt.colorbar()
plt.show()

temp = np.concatenate([Pred[:,:2], Y.reshape(-1,1)],1)
np.savetxt(r"result.txt", temp, fmt='%f', delimiter='   ', newline='\n')
# #================================Precision evaluation=========================
check_coord = df.iloc[Ind,[0,1]].values
D = np.concatenate([Pred[:,:2],Y.reshape(-1,1)],1)
temp = pygmt.surface(data=D, I=I, R=R)
temp = pygmt.grdtrack(check_coord, grid=temp, newcolname='pred')

deep = df.iloc[Ind,2].values
deep_pred = temp[2].values
v = np.abs(deep -  deep_pred)
df = pd.DataFrame(v)
sigma = np.std(v)
rmse = np.sqrt(Mse(deep ,deep_pred))
relative_error = np.abs(v/deep).mean()
df.describe()
