# -*- coding: utf-8 -*-
## Train function ## 

def make_train_step(model, extractor, optimizer, criterion):
    def train_step(x,y):
        
        model.train()
        extractor.eval()

        features = extractor(x)
        pred = model(features)

        #compute loss
        loss = loss_fn(pred,y.float())
        # loss = torchvision.ops.sigmoid_focal_loss(pred, y.float(), reduction = 'mean')
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss, pred
    return train_step
    

### Training and Validation loop function ### 
   
def train_validate(n_epochs, train_fold, val_fold):
    
    # Pre- allocation 
    losses = []
    val_losses = [] 
    epoch_train_losses = []
    epoch_val_losses = []

    early_stopping_tolerance = 5
    early_stopping_counter = 0
    best_auc = 0.0
    
    for epoch in range(n_epochs):
        
        epoch_loss = 0
        true_labels = np.empty((0,))
        pred_labels = np.empty((0,))
        
        
        for i, data in tqdm(enumerate(train_fold), total = len(train_fold)):
                        
            x_batch, y_batch = data 

            x_batch = x_batch.to(device)
                        
            y_batch = y_batch.to(device)

            y_batch = y_batch.unsqueeze(1)  
                        
            # calculate and append losses
            
            loss, pred = train_step(x_batch, y_batch)
            # loss, pred = train_step_foc(x_batch, y_batch)
            epoch_loss += loss/len(train_fold)
            losses.append(loss)
            
            # Append predictions and targets 
            
            pred = (pred.detach().cpu().numpy() >= 0) + 0
            pred_labels = np.append(pred_labels, pred.flatten())
            true_labels = np.append(true_labels, y_batch.cpu().flatten().numpy()
           
        epoch_train_losses.append(epoch_loss)
        
        # Calculate AUC
        epoch_train_auc = sm.roc_auc_score(true_labels, pred_labels)
        
        # Print status
        
        print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))
        print('Train AUC: ', epoch_train_auc)
        
        ### VALIDATION ###
    
        with torch.no_grad():
            
            cum_loss = 0
            true_labels = np.empty((0,))
            pred_labels = np.empty((0,))
            
            for x_batch, y_batch in val_fold: 
                
                x_batch = x_batch.to(device)

                y_batch = y_batch.unsqueeze(1).float()
            
                model.eval()
                
                # If time-series:
                features = extractor(x_batch) # (Batch_size, Seq_length, features)
                yhat = model(features).cpu()
                
                # if static:
                # yhat = model(x_batch).cpu()
                
                val_loss = loss_fn(yhat, y_batch)
                # val_loss = torchvision.ops.sigmoid_focal_loss(yhat, y_batch, reduction = 'mean')
                
                cum_loss += val_loss/len(val_fold)
                val_losses.append(val_loss.item())
                
                pred = (yhat.numpy() >= 0) + 0
                pred_labels = np.append(pred_labels, pred.flatten())
                true_labels = np.append(true_labels, y_batch.flatten().numpy())
            
            # Print status: 
     
            epoch_val_losses.append(cum_loss)
            print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))
            epoch_val_auc = sm.roc_auc_score(true_labels, pred_labels)
            print('Validation AUC: ', epoch_val_auc)
            
            scheduler.step(epoch_val_auc)
        
            best_loss = min(epoch_val_losses)
            
            # Update state_dict based on loss
            if cum_loss <= best_loss:
                # early_stopping_counter = 0
                print("updated best loss")
                loss_best_weights = model.state_dict()
            
            # Update state_dict based on AUC
            if epoch_val_auc > best_auc:
                # early_stopping_counter = 0
                print("updated best AUC")
                best_auc = epoch_val_auc
                AUC_best_weights = model.state_dict()

            # Early stopping

            # early_stopping_counter += 1
        
#             if early_stopping_counter > early_stopping_tolerance:
#                 print("Terminating: early stopping")
#                 break
    
    loss_path = "State_dict/best_loss" 
    torch.save(loss_best_weights, loss_path)
    
    AUC_path = "State_dict/best_AUC" 
    torch.save(AUC_best_weights, AUC_path)
                  
    return epoch_val_auc, epoch_train_auc

## Train and validate ## 

train = trainloader
val = valloader

val_aucs, train_aucs = train_validate(n_epochs, train, val)

