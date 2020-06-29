import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchsummary import summary

import Model
from Dataset import UCF10
from Utils import build_paths


#### Paths #############################################################################################################

class_idxs, train_split, test_split, frames_root, remaining = build_paths()


#### Params ############################################################################################################

print('\n==> Initializing Hyperparameters...\n')


best_acc = 0                                                           # best test accuracy
start_epoch = 0                                                        # start from epoch 0 or last checkpoint epoch
initial_lr = .00001
batch_size = 12
num_workers = 2
num_classes = 101
clip_len = 16
model_summary = False
resume = False
pretrain = False
print_batch = False
nTest = 1

print('Initial Learning Rate: ', initial_lr)
print('Batch Size: ', batch_size)
print('Clip Length: ', clip_len)

#####
# Add: if pretrain == resume: error

### Data ###############################################################################################################

print('\n==> Preparing Data...\n')
'''
trainset = UCF10(class_idxs=class_idxs, split=remaining, frames_root=frames_root,
                 clip_len=clip_len, train=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
'''
trainset = UCF10(class_idxs=class_idxs, split=train_split, frames_root=frames_root,
                 clip_len=clip_len, train=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = UCF10(class_idxs=class_idxs, split=test_split, frames_root=frames_root,
                clip_len=clip_len, train=False)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
'''

testset = UCF10(class_idxs=class_idxs, split=[test_split[0]], frames_root=frames_root,
                clip_len=clip_len, train=False)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
'''
print('Number of Classes: %d' % num_classes)
print('Number of Training Videos: %d' % len(trainset))
print('Number of Testing Videos: %d' % len(testset))


### Model ##############################################################################################################

print('\n==> Building Model...\n')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device Being Used:', device)

######
# Add: (1) if resume: print(...)
######
pretrained_path=None
if pretrain:
    print('Pretrained Weights Loaded From: %s' % pretrained_path)
else:
    print('Model Will Be Trained From Scratch')


model = Model.C3D(num_classes=num_classes, pretrained_path=pretrained_path, pretrained=pretrain)
model = model.to(device)

if model_summary:
    summary(model, input_size=(3, clip_len, 112, 112))

if device == 'cuda':
   model = torch.nn.DataParallel(model)
   cudnn.benchmark = True


### Load Checkpoint or Pretrained Weights ##############################################################################

if resume:
    print('\n==> Resuming from checkpoint...\n')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    model.load_state_dict(checkpoint['model_state'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['opt_dict'])

# state = {
#     'epoch': epoch,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
#     ...
# }
# torch.save(state, filepath)
# To resume training you would do things like: state = torch.load(filepath), and then, to restore the state of each individual object, something like this:
#
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])


### Optimizer, Loss, initial_lr Scheduler ##############################################################################

train_params = [{'params': Model.get_1x_lr_params(model), 'initial_lr': initial_lr},
                {'params': Model.get_10x_lr_params(model), 'initial_lr': initial_lr * 10}]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=initial_lr, momentum=0.9, weight_decay=5e-4)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion.to(device)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


### Training ###########################################################################################################

def train(epoch):
    #print('\n==> Training model...\n')
    start = timer()
    # scheduler.step()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs = nn.Softmax(dim=1)(outputs)
        predicted = torch.max(probs, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if print_batch:
            print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Train]'
                % (epoch+1, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Train]'
                % (epoch+1, train_loss/len(trainloader), 100.*correct/total, current_lr, (end - start)/60))
    # save running checkpoint
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},
                os.path.join('c3d-supervised', 'checkpoint.pth.tar'))

### Testing ############################################################################################################

def test(epoch):
    #print('\n==> Testing model...\n')
    start = timer()
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if print_batch:
                print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Test]'
                    % (epoch+1, batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Test]'
          % (epoch+1, test_loss/len(testloader), 100.*correct/total, current_lr, (end - start)/60))

    # Save checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving Checkpoint..')
        state = {'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
####
# Add in: (1) optim.state, (2) save last 2 checkpoints,
####

for epoch in range(start_epoch, start_epoch+200):
    if epoch == start_epoch:
        print('\n==> Training model...\n')

    train(epoch)

    if (epoch + 1) % nTest == 0:
        test(epoch)

