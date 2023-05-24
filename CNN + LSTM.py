# -*- coding: utf-8 -*-
### Initialize Feature Extractor ### 

class Extractor(nn.Module):
    def __init__(self, state_dict = None):
        super(Extractor, self).__init__()

        # self.extractor = models.resnet18(weights = 'DEFAULT') # if ResNet-18
        self.extractor = models.resnet34(weights = 'DEFAULT') # If ResNet-34

        num_filts = self.extractor.fc.in_features
        self.extractor.fc = nn.Linear(num_filts,1)
        
        # For pre-trained feature extractor: 
        if state_dict:
            self.extractor.load_state_dict(state_dict)

        self.extractor.fc = nn.Identity()
        
    def forward(self, x):
    
        batch_size, time_step, channels, height, width = x.shape
       
        x = x.view(batch_size*time_step, channels, height, width)  # reshape to (batch_size*time_steps, channels, height, width)
        
        features = self.extractor(x)  # extract features
        
        features = features.reshape(batch_size, time_step, 512)
        
        return features





### Initialize LSTM ### 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout = p, batch_first=True)
   
        # Attention:
        # self.attn = nn.Linear(hidden_size, 1) # if attention
        
        # Last FC layers:
        self.fc1 = nn.Linear(hidden_size, 128)        
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
    
        # Pass features through LSTM and make prediction
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        ## If Attention: 
        attn_weights = F.softmax(self.attn(out), dim=1)
        out = torch.sum(attn_weights * out, dim=1)
        
        # out = self.fc1(out[:, -1, :]) # if not attention
        out = self.fc1(out) # If attention
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
  
  
  
  
### Load models ### 

device = "cuda" if torch.cuda.is_available else "cpu"

# state_dict = torch.load("Models_all_runs/extractor_5_steps")
# extractor = Extractor(state_dict = state_dict) # If pre-trained extractor
extractor = Extractor()
extractor.to(device)

# If dropout
# p = 0.5
# model = LSTM(input_size=512, hidden_size=300, num_layers=2, p = p)

# If weight_decay
none = 0
model = LSTM(input_size=512, hidden_size=300, num_layers=1, p = none)

model.to(device)




### initialize hyperparameters ### 

lr = 1e-3
loss_fn = BCEWithLogitsLoss() # Has sigmoid 

# Adam:
optimizer = torch.optim.Adam(model.parameters(), lr = lr) # If dropout
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4) # if weight decay 

# SGD: 
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9) # If dropout
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 1e-4) # if weight decay 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, min_lr = 1e-6)
train_step = make_train_step(model, extractor, optimizer, loss_fn)
# train_step_foc = make_train_step_foc(model, optimizer)


