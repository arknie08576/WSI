import numpy as np
import matplotlib.pyplot as plt

# Define functions

def f(x):
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x

def g(x1, x2):
    return (x1 - 2)**4 + (x2 + 3)**4 + 2 * (x1 - 2)**2 * (x2 + 3)**2

# Define gradients

def gradient_f(x):
    return 40*x**3 + 9*x**2 - 60*x + 10

def gradient_g(x1, x2):
    df_dx1 = 4*(x1 - 2)**3 + 4*(x1 - 2)*(x2 + 3)**2
    df_dx2 = 4*(x2 + 3)**3 + 4*(x2 + 3)*(x1 - 2)**2
    return np.array([df_dx1, df_dx2])

# Gradient Descent Implementation
def gradient_descent(fun, gradient, initial_point, beta, max_iterations=10000, treshold=1e-7):
    x = np.array(initial_point, dtype=float)  # Ensure x is a NumPy array
    path = [x.copy()]
    
    for iteration in range(max_iterations):
        if x.ndim == 0:  # If x is a scalar (1D case)
            gradient_value = gradient(x)
        else:  # If x is a vector (multi-dimensional case)
            gradient_value = gradient(*x)
        
        x -= beta * gradient_value  # Update x
        path.append(x.copy())

        if x.ndim == 0:  # If x is a scalar (1D case)
            if abs(fun(path[-1])-fun(path[-2])) < treshold:
                break
        else:  # If x is a vector (multi-dimensional case)
            if abs(fun(*path[-1])-fun(*path[-2])) < treshold:
                break
    

    return x, path

# 1)

# Plot function f(x)
x_vals = np.linspace(-4, 4, 400)
y_vals = f(x_vals)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')
plt.grid()
plt.legend()
plt.show()

# Plot function g(x1, x2)
x1_vals, x2_vals = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-13, 8, 100))
z_vals = g(x1_vals, x2_vals)
plt.figure(figsize=(8, 5))
plt.contourf(x1_vals, x2_vals, z_vals, levels=50, cmap='viridis')
plt.colorbar(label='g(x1, x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot of g(x1, x2)')
plt.show()


# 2)
# Generate betas
betas = np.array([0.05, 0.01, 0.005, 0.003, 0.001, 0.008, 0.0005, 0.0001, 0.000001])
results_f=[]
results_g=[]
# Generate a set of 10 unique random numbers between -4 and 4
random_set_for_f = set(np.round(np.random.uniform(-4, 4, 10), 2))

# Generate 10 random (x1, x2) points in the given range
random_set_for_g = {(np.round(np.random.uniform(-5, 8), 2), np.round(np.random.uniform(-13, 8), 2)) for iteration in range(10)}


# Let's examine Gradient Descent for given Betas
for beta in betas:


    for initial_point_for_f in random_set_for_f:
        min_f, path_f = gradient_descent(f, gradient_f, initial_point_for_f, beta)

        x_vals = np.linspace(-4, 4, 400)
        y_vals = f(x_vals)
        f_vals = np.array([f(x) for x in path_f])
        results_f.append({
            'Beta': beta,
            'Initial Point': initial_point_for_f,
            'Number of Steps': len(path_f)-1,
            'Result': min_f
        })
        plt.figure(figsize=(8, 5))
        
        
        plt.plot(x_vals, y_vals, label='f(x)')
        plt.plot(path_f, f_vals, 'r.-', label='Gradient Descent Path')
        plt.xlim(-4, 4)  # Set x-axis limits from -4 to 4
        plt.ylim(-100, 3000)  # Set y-axis limits from -100 to 3000
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f"Plot of f(x) with β={beta} and initial point {initial_point_for_f:.2f}")
        plt.grid()
        plt.legend()
        #plt.show()
        plt.show(block=False)  # Show without blocking execution
        #plt.pause(1)  # Wait for 1 second
        plt.close()  # Close the figure

for beta in betas:

    for initial_point_for_g in random_set_for_g:
        min_g, path_g = gradient_descent(g, gradient_g, initial_point_for_g, beta)

        x_vals = np.linspace(-4, 4, 400)
        y_vals = f(x_vals)
        f_vals = np.array([f(x) for x in path_f])
        path_g = np.array(path_g)
        results_g.append({
            'Beta': beta,
            'Initial Point': (float(initial_point_for_g[0]), float(initial_point_for_g[1])),  # Convert tuple values,
            'Number of Steps': len(path_g)-1,
            'Result': min_g
        })
        plt.figure(figsize=(8, 5))
        
        
        plt.contourf(x1_vals, x2_vals, z_vals, levels=50, cmap='viridis')
        plt.plot(path_g[:, 0], path_g[:, 1], 'r.-', label='Gradient Descent Path')
        plt.xlim(-5, 8)  # Set x-axis limits from -5 to 8
        plt.ylim(-13, 8)  # Set y-axis limits from -13 to 8
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar(label='g(x1, x2)')
        #plt.title('Plot of g(x1, x2) with ' + str(beta) + ' as beta and ' + str(initial_point_for_g) + ' as initial point')
        plt.title(f"Plot of g(x1, x2) with β={beta} and initial point ({initial_point_for_g[0]:.2f}, {initial_point_for_g[1]:.2f})")

        plt.grid()
        plt.legend()
        #plt.show()
        plt.show(block=False)  # Show without blocking execution
        #plt.pause(1)  # Wait for 1 second
        plt.close()  # Close the figure
fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

# First table (for results_f)
ax1.axis('tight')
ax1.axis('off')
table_data_f = []
for result in results_f:
    table_data_f.append([result['Beta'], result['Initial Point'], result['Number of Steps'], result['Result']])

# Column labels for table f
column_labels_f = ['Beta', 'Initial Point', 'Number of Steps', 'Result']

# Create the table in ax1
table_f = ax1.table(cellText=table_data_f, colLabels=column_labels_f, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])

# Second table (for results_g)
ax2.axis('tight')
ax2.axis('off')
table_data_g = []
for result in results_g:
    table_data_g.append([result['Beta'], result['Initial Point'], result['Number of Steps'], result['Result']])

# Column labels for table g
column_labels_g = ['Beta', 'Initial Point', 'Number of Steps', 'Result']

# Create the table in ax2
table_g = ax2.table(cellText=table_data_g, colLabels=column_labels_g, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
table_f.auto_set_font_size(False)  # Disable automatic font size adjustment
table_f.set_fontsize(9)  # Set the font size for the table text
table_g.auto_set_font_size(False)  # Disable automatic font size adjustment
table_g.set_fontsize(9)  # Set the font size for the table text
# Adjust layout and display
plt.tight_layout()  # Automatically adjust the subplots to fit into the figure area
plt.show()




