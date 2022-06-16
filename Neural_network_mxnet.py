import mxnet as mx,time,matplotlib.pyplot as plt,os,numpy as np,pandas as pd
from mxnet import nd , autograd,gluon,init
from mxnet.gluon import nn,data as gdata ,loss as gloss, utils as gutils
from copy import deepcopy
import pygmt
from sklearn.metrics import mean_squared_error as Mse

def try_all_gpus():
    '''
    Return all available GPUs, or [mx.cpu()] if there is no GPU.
    :return:
    '''
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

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

#=======================Input Data================================
#Each column of the input data should be: Lon, Lat, Depth, GA, VGG, Wavelength band GA, Wavelength band VGG
Data = read(r"Train_dataset.txt", 7)
d = read(r"Prediction_dataset.txt", 7)
ind = read(r"Ind.txt", 1) #ind.txt is the indices of the check points.

ind = ind.astype(int).tolist()
Ind = []
for i in ind:
    Ind += i

size ,I, R = [361,481], '1m', '140/148/10/16'
#==============================================================
Data_all = pd.DataFrame(np.concatenate([Data,d],0))

Coord = deepcopy(Data_all[[0,1]])
test = Coord.iloc[Ind,:]

label_all = deepcopy(Data_all[2])
features_all = Data_all.drop(columns=[0,1,2])
#=======================Data process================================
#Concatenate the training featrue data and the prediction featrue data for Normalization together.
numerical_features = features_all.dtypes[features_all.dtypes != 'object'].index
features_all[numerical_features] = features_all[numerical_features].apply( 
    lambda x: (x-x.mean())/(x.std()))
features_all[numerical_features]=features_all[numerical_features].fillna(0)

temp1 = features_all.iloc[:len(Data),:]
temp2 = label_all.iloc[:len(Data)]
test_y = temp2[test.index]
test_features = nd.array( temp1.iloc[test.index,:])
test_labels = nd.array( temp2[test.index])

temp1.drop(test.index, inplace=True)
temp2.drop(test.index, inplace=True)
train_features = nd.array(temp1.values)
train_labels = nd.array(temp2.values)

train_data = gdata.ArrayDataset(train_features,train_labels)
test_data = gdata.ArrayDataset(test_features,test_labels)
#=======================Model==================================
class model(nn.Block):
    def __init__(self):
        super(model, self).__init__()
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()

        self.block1.add(
            nn.Dense(16),
            nn.Dense(256),
        )
        self.block2.add(
            nn.Dense(16),
            nn.Dense(256),
        )
        self.block3.add(
            nn.Activation('relu'),
            # nn.ELU(),
            nn.Dense(1)

        )

    def forward(self, input, *args):
        temp = nd.concat(self.block1(input[:,[0,2]]), self.block2(input[:,[1,3]]) , dim=1)
        return self.block3(temp)
        

#==============================Traning========================
def log_rmse(net,features,labels):
    '''
    Return loss for printing.
    :param net: Neural network model
    :param features: Feature data
    :param labels: Label data
    :return: Loss
    '''
    Loss = gloss.L2Loss()
    pred  = nd.clip(net(features.as_in_context(mx.gpu())).abs(),1,float('inf'))
    rmes = (2*Loss(pred,labels.abs().as_in_context(mx.gpu())).mean())

    return  rmes.asscalar()

def _get_batch(batch, ctx):
    '''
    :param batch: Batch size.
    :param ctx: Training context
    :return: features and labels on ctx.
    '''
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    '''Train and evaluate a model.'''
    L,Lt ,Lr= [],[],[]
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l, train_acc_sum, start = [], [], time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        train_l = log_rmse(net, train_data._data[0], nd.array(train_data._data[1]))
        test_l = log_rmse(net, test_data._data[0], nd.array(test_data._data[1]))
        L.append(train_l)
        Lt.append(test_l)
        Lr.append(trainer.learning_rate)
        
        print('epoch %d,train loss %.4f, test loss %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l, test_l,
                 time.time() - start))
    return L,Lt,Lr

def config():
    '''Return parameter for training.'''
    ctx, lr, wd, batch_size, epoch_num = try_all_gpus(), 0.005, 0, 512, 100
    net = model()
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    return net, ctx, lr, wd, batch_size, epoch_num


net,ctx,lr,wd,batch_size,epoch_num=config()

loss = gloss.L2Loss()
lrs=mx.lr_scheduler.CosineScheduler(max_update=int(len(train_labels)/batch_size*80), base_lr=lr, final_lr= lr*0.1, warmup_steps=0)
trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':lr,'wd':wd,'lr_scheduler':lrs})
train_iter = gdata.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_iter = gdata.DataLoader(test_data,batch_size=batch_size,)

a,b,c=train(train_iter, test_iter,net, loss, trainer, ctx, num_epochs=epoch_num)

#==========================Prediction=============================
X = features_all.iloc[len(Data):,:].values
Y = net(nd.array(X).as_in_context(mx.gpu(0)))
Y=Y.asnumpy().reshape(size)
plt.figure()
plt.imshow(Y, vmax=0)
plt.jet()
plt.colorbar()
plt.show()

temp = np.concatenate([d[:,:2], Y.reshape(-1,1)],1)
np.savetxt(r"result.txt", temp, fmt='%f', delimiter='   ', newline='\n')
# ========================Precision evaluation============================
D = np.concatenate([d[:,:2],Y.reshape(-1,1)],1)
temp = pygmt.surface(D[:,0], D[:,1], D[:,2], I=I, R=R)
temp = pygmt.grdtrack(test, grid=temp, newcolname='pred')

deep = test_y
deep_pred = temp.pred.values
v = np.abs(deep -  deep_pred)
df = pd.DataFrame(v)
sigma = np.std(v)
rmse = np.sqrt(Mse(deep ,deep_pred))
relative_error = (v.abs()/deep.abs()).mean()
df.describe()