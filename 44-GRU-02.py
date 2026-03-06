from torch import nn
from d2l import torch as d2l
import tools.nlp as nlp
import matplotlib.pyplot as plt # 用于画图

batch_size, num_steps = 32, 35
train_iter, vocab = nlp.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = nlp.RNNModel(gru_layer, len(vocab))
model = model.to(device)
nlp.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

plt.show()