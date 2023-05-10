# -*- coding: utf-8 -*-
### Evaluation loop ### 

test = testloader

# Load model
mod_state_dict = torch.load("Mod_LSTM_3h/34D_loss_att")
model.load_state_dict(mod_state_dict)
model.eval()

# extractor.eval() # if not already

def testing(model, extractor, dataloader):
    model.eval()
    
    # Pre-allocation
    preds, targets = [], []
    raw_preds = []
    
    with torch.no_grad():    
        for img, target in dataloader:

            img = img.to(device)

            target = target.to(device)

            features = extractor(img) # (Batch_size, Seq_length, features) # if time-series

            prediction = model(features) # If time-series
            
            # prediction = model(img) # if static model
            
            prediction = torch.sigmoid(prediction) # transform output between 0 and 1
            
            prediction = prediction.detach().cpu().numpy()
            target = target.cpu().numpy()
            raw_preds.append(prediction.squeeze(1))
            
            # Set threshold for predictions 
            if prediction < 0.5:
                prediction = 0
            else:
                prediction = 1    

            preds.append(prediction)
            targets.append(target)
        
        # Load metrics
        confusion_matrix = sm.confusion_matrix(targets, preds)
        AUC = sm.roc_auc_score(targets, preds)
        
        return confusion_matrix, AUC
  
  
 ### Evaluate ### 
 
confusion_matrix, AUC, preds, targets = testing(model, extractor, test)
cm_plot = sm.ConfusionMatrixDisplay(np.round((confusion_matrix/(np.sum(confusion_matrix)))*100,2)) # Transform to percentages
cm_plot.plot()
plt.title('Confusion Matrix (%)')
plt.show()

print("AUC-score = ", AUC)
        