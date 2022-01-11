import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import classifier_model as NeuralNet
import pandas as pd
import torch
import warnings
import pickle
import utils

warnings.filterwarnings('ignore') # Ignore mgz image warnings --> uncomment to see output
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Test to see if GPU is available
print(device)

torch.cuda.empty_cache() # Empty cuda cash to get more memory 

splits = 1


for l in range(splits):
    
    data = pd.read_csv('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/csv_data/alz_data.csv')
    df_train, df_test = utils.train_test_split(data)

    X_train, X_test = df_train.Path.astype(str).tolist(), df_test.Path.astype(str).tolist()
    y_train, y_test = df_train.Label.tolist(), df_test.Label.tolist()

    dataset_train = utils.mydataLoader(X_train, y_train, 0) # Class needs paths, label, train or val as binary
    dataset_val   = utils.mydataLoader(X_test,  y_test,  1)

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
    dataloader_val   = DataLoader(dataset_val,   batch_size=1, shuffle=True, num_workers=4)

    print(f'Length of training data: {len(dataloader_train)}')
    print(f'Length of validation data: {len(dataloader_val)}')

    learning_rates = [2*0.0001] 
    criterion = nn.CrossEntropyLoss()

    for m in range(1):

        my_nn = NeuralNet.ConvNet().to(device) # Create instance of model

        optimizer = torch.optim.Adam(my_nn.parameters(), learning_rates[0])

        count = 0
        losses        = []
        accepoch      = []
        accuracyVal   = []
        accuracyTrain = []
        lossepoch     = []
        v_lossepoch   = []

        load = True #Change here to load model --> remember to change dict and model path
        if load:
            path_to_model = '/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/models/3rd_run_new_best-model_epoch_110.pth.tar'
            my_nn.load_state_dict(torch.load(path_to_model), strict=False )
            losses_dic = pickle.load(open('/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/models/3rd_run_dictionary_CNN.p' , "rb"))  
            count = losses_dic['count']
            losses = losses_dic['losses']
            accepoch  = losses_dic['accepoch']
            accuracyVal  = losses_dic['accuracyVal']
            accuracyTrain  = losses_dic['accuracyTrain']
            lossepoch  = losses_dic['lossepoch']
            v_lossepoch  = losses_dic['v_lossepoch']

        start_from = count 
        for epoch in range(start_from, 1600):  # loop over the dataset multiple times
            train_correct  = 0
            val_correct    = 0
            total_val      = 0
            total_train    = 0
            running_loss   = 0.0
            v_running_loss = 0.0
            
            my_nn.train()
            for i, (data, target) in enumerate(dataloader_train):
                inputs, labels = data.to(device), ((torch.tensor(target))).to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = my_nn(inputs)
                
                loss = criterion(outputs, labels)
                losses.append(loss) # Append loss to list for plotting
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_train += len(target) # Number of training imgs passed through model
             
                
                predicted =  torch.max(outputs.data, 1)[1]

                train_correct += (labels.squeeze() == predicted.squeeze()).sum().item()

            my_nn.eval()
            for j, (v_images, v_labels) in enumerate(dataloader_val):
                val = v_images.to(device)
                v_labels  = (torch.tensor(v_labels)).to(device)

                # Forward 
                v_outputs = my_nn(val)
                total_val += len(v_labels)
                v_loss = criterion(v_outputs, v_labels)
                v_running_loss += v_loss.item()
                
                # Get predictions from the maximum value
                v_predicted =  torch.max(v_outputs.data, 1)[1]

                val_correct += (v_predicted.squeeze() == v_labels.squeeze()).sum().cpu().item()

            losses_dic = {}
            count += 1 
            names = [
                'count',
                'losses',
                'accepoch',
                'accuracyVal',
                'accuracyTrain',
                'lossepoch',
                'v_lossepoch'
                    ]

            lists = [
                count,
                losses,
                accepoch,
                accuracyVal,
                accuracyTrain,
                lossepoch,
                v_lossepoch
            ]

            for name, lsts in zip(names, lists):
                losses_dic[name] = lsts

            if ((epoch % 10 == 0) and (epoch > 100)):
                torch.save(my_nn.state_dict(), f"/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/models/new_best-model_epoch_{epoch}.pth.tar")
                pickle.dump(losses_dic, open(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/models/dictionary_CNN_{epoch}.p' , "wb"))    #s
#            print('Epoch: {}  Train_Loss: {}  Val_Loss: {} Train Accuracy: {} % Val Accuracy: {} %'.format(epoch, running_loss/len(dataloader_train), v_running_loss/len(dataloader_val), (100*train_correct)/total_train, (100*val_correct)/total_val))
            t_loss, v_loss, t_acc, v_acc = running_loss/len(dataloader_train), v_running_loss/len(dataloader_val), (train_correct)/total_train, (val_correct)/total_val
            print(f'Epoch: {epoch} Training Loss: {t_loss:.4f} Validation Loss {v_loss:.4f} Training Accuracy {t_acc:.2%} Validation Accuracy {v_acc:.2%}')
            
            # Append losses, accuracies, iterations 
            lossepoch.append(running_loss/len(dataloader_train))
            v_lossepoch.append(v_running_loss/len(dataloader_val))
            accuracyVal.append(val_correct/total_val)      
            accuracyTrain.append(train_correct/total_train)      

            # Loss plot
            plt.figure()
            tloss, = plt.plot(lossepoch)
            vloss, = plt.plot(v_lossepoch)
            tloss.set_label('Training Loss')
            vloss.set_label('Validation loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("CNN Training & Validation Loss")
            plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/lr_{learning_rates[0]}_loss_curve.png')    
            plt.clf()
            plt.cla()
            plt.close()

            # Accuracy plot
            plt.figure()
            tloss, = plt.plot(accuracyTrain)
            vloss, = plt.plot(accuracyVal)
            tloss.set_label('Training Accuracy')
            vloss.set_label('Validation Accuracy')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0,1)
            plt.legend()
            plt.title("CNN Training & Validation Accuracy")
            plt.savefig(f'/dtu-compute/ADNIbias/AlzPred_Oskar_Anders/git_code/AlzPred/classifier/plots/lr_{learning_rates[0]}_accuracy_curve.png')
            plt.clf()
            plt.cla()
            plt.close()

        #torch.save(my_nn.state_dict(), "new_best-model_learningrate_" + str(l)  )
print('Finished Training')