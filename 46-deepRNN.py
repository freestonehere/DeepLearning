from torch import nn
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图

batch_size, num_steps = 32, 35
train_iter, vocab = nlp.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = nlp.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
nlp.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)

plt.show()