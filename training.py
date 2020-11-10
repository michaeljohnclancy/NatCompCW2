from psotorch import PSO

def run_training_cycle(x_train, y_train, model, loss, num_epochs, inertia, a1, a2, population_size, device=None, verbose=False):

    cuda = device is not None
    if cuda:
        model = model.to(device)
        loss = loss.to(device)

    optimizer = PSO(x_train, y_train, model=model, loss=loss, dim=56, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=1, cuda=cuda)

    for i in range(num_epochs):

        y_train_preds = model(x_train)
        fitness = loss(y_train_preds, y_train)

        class_classified = (y_train_preds>0.5).float()
        accuracy = sum(y_train[i] == class_classified[i] for i in range(len(class_classified)))/y_train_preds.shape[0]

        if verbose:
            print(f"Epoch {i}: Fitness = {fitness}; Acc = {accuracy}")

        optimizer.step()

    return model, fitness, y_train_preds
