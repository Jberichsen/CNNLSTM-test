# -*- coding: utf-8 -*-
### Initialize model and add drop-out to model ###

model = models.resnet18(weights = 'DEFAULT')

## Change input channels to 1 instead of 3 if 1 channel
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding = (3, 3), bias=False)

# Add new final layer

num_filts = model.fc.in_features
model.fc = nn.Linear(num_filts,1)

# Freeze all but layer 4 and last

for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

#Add spatial dropout - if not weight_decay

p = 0.5
model.layer1[0].add_module('Dropout', nn.Dropout2d(p = p))
model.layer1[1].add_module('Dropout', nn.Dropout2d(p = p))

model.layer2[0].add_module('Dropout', nn.Dropout2d(p = p))
model.layer2[1].add_module('Dropout', nn.Dropout2d(p = p))

model.layer3[0].add_module('Dropout', nn.Dropout2d(p = p))
model.layer3[1].add_module('Dropout', nn.Dropout2d(p = p))

model.layer4[0].add_module('Dropout', nn.Dropout2d(p = p))
model.layer4[1].add_module('Dropout', nn.Dropout2d(p = p))





### Initialize hyperparameters ### 

lr = 1e-3
loss_fn = BCEWithLogitsLoss() # Has sigmoid     

# Adam:
optimizer = torch.optim.Adam(model.parameters(), lr = lr) # If dropout
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4) # if weight decay 

# SGD 
optimizer = torch.optim.SGD(model.parameters(), lr = lr) # If dropout
optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = 1e-4) # if weight decay 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr = 1e-6)
train_step = make_train_step(model, optimizer, loss_fn)
# train_step_foc = make_train_step_foc(model, optimizer)
n_epochs = 20





