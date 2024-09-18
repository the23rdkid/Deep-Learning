
'''
Naomi Maranga
ESE 546 HW3
Due Date: Nov 1, 1159h
'''
#wall of imports
import os
import numpy as np
import torch
import torch.optim
import torch.nn

#download data
text = "./book-war-and-peace.txt"
if not os.path.exists(text):
  os.system("wget https://www.gutenberg.org/ebooks/100"
                 "https://www.gutenberg.org/cache/epub/2600/pg2600.txt")

with open(text, "r") as fp:
  #read in text
  content = fp.readlines()

content = ''.join(content)
#replace newline chars with white space
content = content.replace('\n', '')

#make a set of list of characters
chars = list(set(content))
#sort the list of chars
chars = sorted(chars)
#get the length of chars used in war and peace
num_chars = len(chars)
#some print statements to visualise what we've done so far
print("length of characters:", num_chars)
print("individual sorted characters list, both cases:", chars)

#get the counts of character occurrence
def get_counts(text, chars):
    char_count = {}
    for ch in chars:
        char_count[ch] = 0
    for ch in text:
        char_count[ch] += 1
    print(char_count)


print("No. of times individual characters appear in text")
get_counts(text, chars)
'''
output of above lines of code:
length of characters: 81
individual sorted characters list, both cases: [' ', '!', '"', "'", '(', ')', '*',
',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=',
'?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
  'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'ä', 'é', 'ê']
No. of times individual characters appear in text
{' ': 0, '!': 0, '"': 0, "'": 0, '(': 0, ')': 0, '*': 0, ',': 0, '-': 3, '.': 2,
'/': 1, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,
'9': 0, ':': 0, ';': 0, '=': 0, '?': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0,
'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0,
'Z': 0, 'a': 3, 'b': 1, 'c': 1, 'd': 1, 'e': 2, 'f': 0, 'g': 0, 'h': 0, 'i': 0,
'j': 0, 'k': 1, 'l': 0, 'm': 0, 'n': 1, 'o': 2, 'p': 1, 'q': 0, 'r': 1, 's': 0,
't': 2, 'u': 0, 'v': 0, 'w': 1, 'x': 1, 'y': 0, 'z': 0, 'à': 0, 'ä': 0, 'é': 0, 'ê': 0}
'''

#Q4.a)
#one-hot encoding
one_hot_map = {}
for idx, ch in enumerate(chars):
  one_hot_map[ch] = idx

def get_vector(text):
  indices = [one_hot_map[ch] for ch in text]
  vectorised = np.eye(num_chars)[indices]
  return vectorised, indices

#convert text to vec
vectorised_text, index_text = get_vector(content)
vectorised_text = torch.Tensor(vectorised_text)
index_text = torch.Tensor(index_text).long()

print(len(vectorised_text))

num_train = 500000 #chose an absurdly big number for the number of my training samples
num_val = 5000 + num_train #choosing my validation set to be slightly larger than my training set at the risk of slightly higher bias
x_train = vectorised_text[:num_train]
y_train = index_text[:num_train]

x_val = vectorised_text[num_train:num_val]
y_val = index_text[num_train:num_val]

#return batches and use gpu for faster computation
def get_batch(counter, gpu, text, labels):
  x_batch = text[counter * 32: (counter + 1) * 32]
  y_batch = labels[counter * 32 + 1: (counter + 1) * 32]
  x_batch = x_batch.reshape(-1, num_chars)
  ln = min(len(x_batch), len(y_batch))
  x_batch, y_batch = x_batch[:ln], y_batch[:ln]
  if gpu:
    x_batch = x_batch.cuda()
    y_batch = y_batch.cuda()
  return(x_batch, y_batch)

#Q4.b)
x, y = get_batch(1, False, vectorised_text, index_text)
#check the shape of x and y
print("x_shape", x.shape)
print("target", y)

#training parameters
num_batches = len(y_train)//32
num_val_batches = len(y_val)//32
gpu_1 = torch.cuda.is_available()
learning_rate = 0.001 #earlier had this as 0.003 facepalm! double checked what 10^-3 was and boy do i feel silly! :p
timesteps = 15
hidden_size = 200
cosine = True
'''
shape of our matrices:
x_shape torch.Size([31, 81])
target tensor([51, 64, 54,  0, 36, 71, 53, 53, 51,  0, 51, 68, 55,  0, 64, 65, 73,  0,
        60, 71, 69, 70,  0, 56, 51, 63, 59, 62, 75,  0, 55])
'''

class charRNN(torch.nn.Module):
  def __init__(self, hidden_dim, out_dim):
    super(charRNN, self).__init__()
    #initialise an RNN with one hidden layer
    self.rnn = torch.nn.RNN(input_size = num_chars,
                            hidden_size = hidden_dim,
                            num_layers = 1,
                            nonlinearity = "tanh",
                            bias = True
                            )
    #add an embedding layer
    self.embed = torch.nn.Linear(num_chars, num_chars)
    self.linear = torch.nn.Linear(hidden_dim, out_dim)
    self.hidden_dim = hidden_dim

  #forward pass
  def forward(self, x, hidden):
    embed = self.embed(x)
    embed = embed.view(-1, 1, num_chars)
    out, hidden = self.rnn(embed, hidden)
    out = out.view(-1, self.hidden_dim)
    out = self.linear(out)
    return out, hidden

def unroll(steps = 32, start = 0, sample = False):
  hidden = torch.zeros(1, 1, hidden_size)
  net.eval()
  #initialise with random chars
  data = x_train[start].reshape(1, 1, -1)
  prediction = '' + chars[y_train[start]]
  for s in range(steps):
    hidden = hidden.cpu().detach()
    if gpu_1:
      hidden = hidden.cuda()
      data = data.cuda()
    output, hidden = net(data, hidden)
    if sample:
      output = torch.nn.functional.softmax(output, dim = 1)
      output = np.reshape(output.cpu().detach().numpy(), (-1))
      #get a random character
      out_char = np.random.choice(list(range(num_chars)), p = output)
    else:
      out_char = torch.argmax(output)

    prediction = prediction + chars[out_char]
    data = torch.zeros(1, 1, num_chars)
    data[0, 0, out_char] = 1
  print('unrolling RNN: %s' %prediction)

net = charRNN(hidden_size, num_chars)
if gpu_1:
  net = net.cuda()
  print("training RNN using GPU")
else:
  print("training RNN using CPU")
optim = torch.optim.Adam(net.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones = [10, 20, 25], gamma = 0.2)

for e in range(timesteps):
  print('timestep')
  hidden = torch.zeros(1, 1, hidden_size)
  train_loss = 0.0
  for i in range(num_batches):
    net.train()
    hidden = hidden.cpu().detach()
    if gpu_1:
      hidden = hidden.cuda()
    optim.zero_grad()
    #train the model and make recurrent updates
    data, target = get_batch(i, gpu_1, x_train, y_train)
    output, hidden = net(data, hidden)
    loss = criterion(output, target)
    loss.backward()
    #clip gradients before optim.step()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    #update steps
    optim.step()
    train_loss += loss.item()

    #accumulate training statistics
    i = num_batches * e + i
    if i% 1000 == 0:
      train_loss = train_loss/1000
      print('loss at %d: %f' % (i, train_loss))
      train_loss = 0.0
      unroll(100, 0)

    #evaluate the model every 1000 steps
    if i % 1000 == 0:
      net.eval()
      hidden_val = torch.zeros(1,1, hidden_size)
      total_loss = 0.0
      total_acc = 0.0
      #finding validation scores
      for t_i in range(num_val_batches):
        hidden_val = hidden_val.cpu().detach()
        if gpu_1:
          hidden_val = hidden_val.cuda()
        val_data, val_target = get_batch(t_i, gpu_1, x_val, y_val)
        output, hidden_val = net(val_data, hidden_val)
        loss = criterion(output, val_target)
        total_loss += loss.item()
        out_val = np.reshape(output.cpu().detach().numpy(), (-1, num_chars))
        total_acc = np.mean(np.argmax(out_val, axis = 1)==val_target.cpu().numpy())
        total_loss = total_loss/num_val_batches
        total_acc = total_acc/num_val_batches * 100
        print("validation loss at %d: %f %f" % (i, total_loss, total_acc))

  if cosine:
    lr_scheduler.step()


unroll(500, 0, False)
print("######")
unroll(500, 0, True)
print('######')
unroll(500, 0, True)
print('######')
unroll(500, 0, True)
torch.save(net, 'net.pth')

#train a self attention based neural network for this same problem
