import matplotlib.pyplot as plt
from pecpy.core import *

# Load resources.
cheese_mini = get_resource("cheese_mini.png")
# Set parameters.
alpha, eta = 30, 0.25


def make_plot(x):
    plt.subplot(2, 2, 1)
    plt.title("Design field")
    plt.imshow(x["design"], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("Dose field")
    plt.imshow(x["dose"], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title("Physical field")
    plt.imshow(x["phys"], cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title("Error")
    plt.imshow(x["error"], cmap='gray')
    plt.tight_layout()


# Develop dose (no PEC).
x = develop_dose(cheese_mini, alpha, eta, design=cheese_mini)
# Visualize the result.
plt.figure(figsize=(5, 5))
make_plot(x)

# Optimize the dose.
betas = [2.0 ** i for i in np.arange(0, 13)]
solver = LBFGSB(5e-7, 1e8 * np.finfo(float).eps, 50, disp=1)
stepper = FilterProjectionStepper(10, [0.1])
output = DirectoryOutput("example", overwrite=True)
x = optimize_dose(cheese_mini, alpha, eta, betas, stepper, solver, output=output)
# Visualize the result.
plt.figure(figsize=(5, 5))
make_plot(x)

plt.show()
