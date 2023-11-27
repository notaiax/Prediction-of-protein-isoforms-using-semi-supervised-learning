import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

NUM_FEATURES = 18965
NUM_OUTPUT = 156958

# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        activation_fn = nn.ReLU
        num_hidden = (NUM_FEATURES + NUM_OUTPUT) // 2
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, num_hidden),
            activation_fn(),
            nn.Linear(num_hidden, NUM_OUTPUT),
            activation_fn()
        )

    def forward(self, x):
        return self.net(x)

model = Net()
device = torch.device('cuda')  # use cuda or cpu
model.to(device)
print(model)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# # Test the forward pass with dummy data
# out = model(torch.randn(2, 3, 32, 32, device=device))
# print("Output shape:", out.size())
# print(f"Output logits:\n{out.detach().cpu().numpy()}")
# print(f"Output probabilities:\n{out.softmax(1).detach().cpu().numpy()}")








# batch_size = 64
# num_epochs = 10
# validation_every_steps = 500

# step = 0
# model.train()

# train_accuracies = []
# valid_accuracies = []

# for epoch in range(num_epochs):

#     train_accuracies_batches = []

#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Forward pass, compute gradients, perform one training step.
#         # Your code here!

#         # Forward pass.
#         output = model(inputs)

#         # Compute loss.
#         loss = loss_fn(output, targets)

#         # Clean up gradients from the model.
#         optimizer.zero_grad()

#         # Compute gradients based on the loss from the current batch (backpropagation).
#         loss.backward()

#         # Take one optimizer step using the gradients computed in the previous step.
#         optimizer.step()

#         # Increment step counter
#         step += 1

#         # Compute accuracy.
#         predictions = output.max(1)[1]
#         train_accuracies_batches.append(accuracy(targets, predictions))

#         if step % validation_every_steps == 0:

#             # Append average training accuracy to list.
#             train_accuracies.append(np.mean(train_accuracies_batches))

#             train_accuracies_batches = []

#             # Compute accuracies on validation set.
#             valid_accuracies_batches = []
#             total_loss = 0

#             with torch.no_grad():
#                 model.eval()
#                 for inputs, targets in test_loader:
#                     inputs, targets = inputs.to(device), targets.to(device)
#                     output = model(inputs)
#                     loss = loss_fn(output, targets)
#                     total_loss += loss

#                     predictions = output.max(1)[1]

#                     # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
#                     valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

#                 model.train()

#             # Append average validation accuracy to list.
#             valid_accuracies.append(np.sum(valid_accuracies_batches) / len(test_set))

#             print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
#             print(f"             test accuracy: {valid_accuracies[-1]}")
#             print(f"             total loss: {total_loss}")

# print("Finished training.")