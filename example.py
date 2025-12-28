import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set_theme(style="whitegrid")
from tensor import Tensor
from function import ReLU, CrossEntropyWithSoftmax
from optimizer import SGD, Adam
from module import Module, Linear


def create_spiral_data(points_per_class, num_classes):
    X = np.zeros((points_per_class * num_classes, 2))
    y = np.zeros(points_per_class * num_classes, dtype='uint8')
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
    return X, y

def create_moons_data(points_per_class=100, noise=0.1):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=points_per_class*2, noise=noise, random_state=42)
    return X.astype(np.float32), y.astype(np.uint8)

def create_circles_data(points_per_class=100, noise=0.1, factor=0.5):
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=points_per_class*2, noise=noise, factor=factor, random_state=42)
    return X.astype(np.float32), y.astype(np.uint8)

def choose_dataset_console():
    print("Select dataset:")
    print("1. Spiral")
    print("2. Moons")
    print("3. Circles")
    
    choice = input("Enter 1, 2, or 3: ").strip()
    points_per_class = int(input("Enter number of points per class (e.g., 100): ").strip() or "100")
    noise = float(input("Enter noise level (e.g., 0.1): ").strip() or "0.1")
    
    if choice == "1":
        num_classes = int(input("Enter number of classes for spiral (e.g., 3): ").strip() or "3")
        X, y = create_spiral_data(points_per_class, num_classes)
    elif choice == "2":
        print("Moons dataset always has 2 classes.")
        X, y = create_moons_data(points_per_class, noise)
    elif choice == "3":
        print("Circles dataset always has 2 classes.")
        X, y = create_circles_data(points_per_class, noise)
    else:
        print("Invalid choice, defaulting to Spiral with 3 classes.")
        num_classes = 3
        X, y = create_spiral_data(points_per_class, num_classes)
    
    return X, y


def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k', alpha=0.8)
    plt.title("Generated Spiral Dataset", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = np.argmax(model.predict(Tensor(grid_points, requires_grad=False)), axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Model Decision Boundary", fontsize=16)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    
def animate_decision_boundary(model, X, y, history, filename="decision_boundary_evolution.gif"):
    """
    Creates and saves an animation of the decision boundary evolving over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    epochs = sorted(history.keys())
    
    def update(frame):
        ax.clear()
        epoch = epochs[frame]
        
        model.set_parameters(history[epoch])

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        Z = np.argmax(model.predict(Tensor(grid_points, requires_grad=False)), axis=1)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
        ax.set_title(f"Decision Boundary at Epoch {epoch}", fontsize=16)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        return ax.collections + ax.patches + ax.texts

    ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=100, blit=False)
    
    print(f"\nSaving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=10)
    print("Animation saved.")
    plt.close(fig)

class MLP(Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        hidden = 100
        self.fc1 = Linear(num_features, hidden)
        self.fc2 = Linear(hidden, num_classes)

    def forward(self, X: Tensor):
        h = ReLU.apply(self.fc1(X))      
        out = self.fc2(h)
        return out

    def predict(self, X):
        return self.forward(X).data
    
if __name__ == "__main__":
    print("Generating dataset...")
    POINTS_PER_CLASS = 100
    NUM_CLASSES = 3
    X, y = choose_dataset_console()


    print("Data generated.")

    print("Visualizing the raw data...")
    plot_data(X, y)

    model = MLP(num_features=2, num_classes=NUM_CLASSES)

    print("Demonstrating the decision boundary for the UNTRAINED model.")
    plot_decision_boundary(model, X, y)

    print("\nStarting training...")

    X_tensor = Tensor(X)

    num_samples = len(y)
    y_one_hot = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)
    y_one_hot[np.arange(num_samples), y] = 1
    y_true = Tensor(y_one_hot, requires_grad=False)

    # 1) Vanilla SGD:
    #optimizer = SGD(model.parameters(), lr=0.1)

    # 2) SGD with momentum:
    #optimizer = SGD(model.parameters(), lr=0.05, momentum=0.8)

    # 3) Adam:
    optimizer = Adam(model.parameters(), lr=1e-3)

    parameter_history = {}

    EPOCHS = 10000
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        y_pred = model.forward(X_tensor)
        loss = CrossEntropyWithSoftmax.apply(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS}, Loss: {loss.data.item():.4f}")
            parameter_history[epoch] = [p.data.copy() for p in model.parameters()]

    print("Training complete.")

    animate_decision_boundary(model, X, y, parameter_history)

    print("\nVisualizing the final trained model's decision boundary...")
    plot_decision_boundary(model, X, y)
