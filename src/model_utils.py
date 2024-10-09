import torch


def train_epoch(model, loader, optimizer, loss_fn, reporting_freq=1000) -> float:
    """Train a single epoch (should be multiuse and multipurpose)

    Args:
        model (torch.nn.Model): trained model
        loader (torch.utils.data.DataLoader): train dataset as a dataloader object
        optimizer (torch.optim.optimizer): selected model optimizer object
        loss_fn (torch.nn.loss_fn): selected loss function object
        reporting_freq (int, optional): Frequency of loss reporting (in batches). Defaults to 1000.

    Returns:
        float: training loss
    """
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(loader):

        # Unpack tuple
        inputs, labels = data
        # Zero the gradients
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % reporting_freq == reporting_freq-1:
            # Average the loss per batch
            last_loss = running_loss / reporting_freq 
            print(f'  batch {i+1} loss: {last_loss:2f}')
            # Reset
            running_loss = 0.

    return running_loss / (i + 1)


def val_epoch(model, loader, loss_fn) -> float:
    """Validate a single epoch (should be multiuse and multipurpose)

    Args:
        model (torch.nn.Model): trained model
        loader (torch.utils.data.DataLoader): val dataset as a dataloader object
        loss_fn (torch.nn.loss_fn): selected loss function object

    Returns:
        float: validation loss
    """
    running_loss = 0

    with torch.no_grad():
        for i, data in enumerate(loader):

            # Unpack tuple
            inputs, labels = data
            # Predict
            outputs = model(inputs)
            # Compute the loss
            loss = loss_fn(outputs, labels)
            # Gather data
            running_loss += loss

    return running_loss / (i + 1)

