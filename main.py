from digits.core.mlp import MLP

# Hyper-parameters
increment = 0.05
interations = 100

# Create network
x = [2, 3, -1]
mlp = MLP(3, [10, 10, 10, 1])

# Instantiate parameters
xs = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
ys = [1, -1, -1, 1]


for iter in range(interations):
    # Forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    if loss == 0:
        break

    # Backward pass
    for p in mlp.parameters():
        p.grad = 0
    loss.backward()

    # Update (gradient decent)
    for p in mlp.parameters():
        learning_rate = (increment * 1.0) - (increment * 0.9 * (iter / interations))
        p.data += -learning_rate * p.grad

    print(iter, ":", loss.data)

print(ypred)
