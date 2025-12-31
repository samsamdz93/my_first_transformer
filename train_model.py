import torch
import torch.nn as nn
from time import time
from numpy import inf
import json

# Function to train the model
def train_model(model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    nepochs,
    modulo = 1,
    path_logs = None,
    path_model = None,
    device = None,
    VOID_TOKEN = None):
    
    if device is None:
        device = 'cpu'
    # Save loss, accuracy and training time
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    total_time = 0

    for epoch in range(nepochs):
        time_start = time()
        train_loss = 0.
        test_loss = 0.

        ###################
        # Train the model #
        ###################
        model.train()

        # Useful to compute accuracy
        good_predictions = 0
        total_predictions = 0
        dataset_size = 0
        cpt_batch = 0
        loss = 0

        for _, (data, target) in enumerate(train_loader):
            # Counting size of dataset
            batch_size, max_len = target.shape
            dataset_size += batch_size
            
            # Put the data on the appropriate device
            data = data.to(device = device)
            target = target.to(device = device)

            # Getting the input and the label of the model
            inp = target[:, :-1]
            label = target[:, 1:]

            # Checking if there are only padding
            mask = (label != VOID_TOKEN)
            if len(label[mask]) != 0:
                cpt_batch += mask.int().sum()
                
                # Computation of the output
                output = model(data, inp)
    
                # Computation of the loss
                current_loss = criterion(output.transpose(1,2), label)
                loss += current_loss
                train_loss += current_loss.item()
    
                # Computation of the predictions
                predictions = torch.argmax(output, dim = -1)
                predictions = (predictions[mask] == label[mask]).int()
    
                # Counting the good predictions
                good_predictions += predictions.sum()
                total_predictions += mask.int().sum()
    
                # Making an optimizer step
                if cpt_batch >= batch_size :
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    cpt_batch = 0
                    loss = 0

        # Computing training accuracy and normalizing train loss
        train_accuracy = float(good_predictions.float()/total_predictions)
        train_loss /= total_predictions

        ##################
        # Test the model #
        ##################
        model.eval()

        # Useful to compute accuracy
        good_predictions = 0
        total_predictions = 0
        dataset_size = 0

        for _, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                # Counting size of dataset
                batch_size, max_len = target.shape
                dataset_size += batch_size
                
                # Put the data on the appropriate device
                data = data.to(device = device)
                target = target.to(device = device)
    
                # Getting the input and the label of the model
                inp = target[:, :-1]
                label = target[:, 1:]
    
                # Checking if there are only padding
                mask = (label != VOID_TOKEN)
                if len(label[mask]) != 0:
                    cpt_batch += mask.int().sum()
                    
                    # Computation of the output
                    output = model(data, inp)
        
                    # Computation of the loss
                    current_loss = criterion(output.transpose(1,2), label)
                    loss += current_loss
                    test_loss += current_loss.item()
        
                    # Computation of the predictions
                    predictions = torch.argmax(output, dim = -1)
                    predictions = (predictions[mask] == label[mask]).int()
        
                    # Counting the good predictions
                    good_predictions += predictions.sum()
                    total_predictions += mask.int().sum()

        # Computing test accuracy and normalizing test loss
        test_accuracy = float(good_predictions.float()/total_predictions)
        test_loss /= total_predictions

        ###################
        # Storing results #
        ###################
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        total_time += time() - time_start


        # Saving the logs
        d = {
            "Epoch" : epoch,
            "Training Loss" : train_loss.item(),
            "Validation Loss" : test_loss.item(),
            "Training Acc" : train_accuracy,
            "Validation Acc" : test_accuracy,
            "Time" : time() - time_start,
            "Total time" : total_time
        }

        if path_logs is not None:
            with open(path_logs, "a") as f:
                json.dump(d, f)
                f.write('\n')

        # Saving the model
        if path_model is not None:
            torch.save(model.state_dict(), path_model)

        # Log the results
        if epoch % modulo == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f} \tTraining acc: {:.6f} \tTest acc: {:.6f}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))
    return train_losses, test_losses, train_accuracies, test_accuracies, total_time






