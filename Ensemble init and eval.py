# -*- coding: utf-8 -*-
# Initialize test_set
test = testloader

# Initialize models

AUC_state_dict_1 = torch.load("Cross_val/Fold_1_AUC_att")
model_1.load_state_dict(AUC_state_dict_1)
model_1.eval()

AUC_state_dict_2 = torch.load("Cross_val/Fold_3_AUC_att")
model_2.load_state_dict(AUC_state_dict_2)
model_2.eval()

AUC_state_dict_3 = torch.load("Cross_val/Fold_5_AUC_att")
model_3.load_state_dict(AUC_state_dict_3)
model_3.eval()

AUC_state_dict_4 = torch.load("Cross_val/Fold_1_AUC_2nd")
model_4.load_state_dict(AUC_state_dict_4)
model_4.eval()

AUC_state_dict_5 = torch.load("Cross_val/Fold_2_AUC_2nd")
model_5.load_state_dict(AUC_state_dict_5)
model_5.eval()

# If not already
extractor.eval()

models = [model_1, model_2, model_3, model_4, model_5]

def testing(models, dataloader):
    preds, targets = [], []
    
    with torch.no_grad():    
        for img, target in dataloader:
            vote_neg, vote_pos = 0, 0
            img = img.to(device)
            target = target.to(device)
            features = extractor(img)
            
            for model in models:
                prediction = torch.sigmoid(model(features))
                prediction = prediction.detach().cpu().numpy()
                
                if prediction < 0.5:
                    vote_neg += 1
                else:
                    vote_pos += 1
            
            target = target.cpu().numpy()
            
            if vote_neg > vote_pos:
                prediction = 0
            else:
                prediction = 1
            
            preds.append(prediction)
            targets.append(target)
            
        confusion_matrix = sm.confusion_matrix(targets, preds)
        AUC = sm.roc_auc_score(targets, preds)
        
        return confusion_matrix, AUC

## Plot results ## 

confusion_matrix, AUC = testing(models, test)
# cm_plot = sm.ConfusionMatrixDisplay(confusion_matrix)
cm_plot = sm.ConfusionMatrixDisplay(np.round((confusion_matrix/(np.sum(confusion_matrix)))*100,2))
cm_plot.plot()
plt.title('Confusion Matrix (%)')
plt.show()

print("AUC-score = ", AUC)
