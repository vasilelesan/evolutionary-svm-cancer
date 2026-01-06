import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
    # load dataset from sklearn
    data = load_breast_cancer()
    x = data.data
    y = data.target
    
    # transform labels to -1 and 1 for svm
    y = np.where(y == 0, -1, 1)
    
    # scale data for better convergence
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # split into train and test sets
    #20% used for testing, 80% used for training
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    print("data loaded successfully")
    print("train shape:", x_train.shape)

    #Compute Q matrix necessary for Fitness Function using @ operator
    print("Compute Q:")
    y_train_col=y_train.reshape(-1,1)
    Q=(y_train_col @ y_train_col.T) * (x_train @x_train.T)

    return x_train, x_test, y_train, y_test, Q

def fitness_function(alpha, Q):
    return np.sum(alpha) - 0.5 * (alpha.T @ Q @ alpha)

# function for vectors to respect box constraint and equality constraint
def adjustment_function(alpha, y_train, C=1.0):
    # sorting max and min values to be bigger than 0, resp smaller than C
    alpha[alpha < 0] = 0
    alpha[alpha > C] = C

    #compute error
    err = np.sum(alpha * y_train)

    #find indexes for +1 class and -1 class
    idx_plus = np.where(y_train == 1)[0]
    idx_minus = np.where(y_train == -1)[0]

    #adjustment 
    if err > 0:
        adjustment = err / len(idx_plus)
        alpha[idx_plus] = alpha[idx_plus] - adjustment
    else:
        adjustment = abs(err) / len(idx_minus)
        alpha[idx_minus] = alpha[idx_minus] - adjustment

     # final sorting max and min values to be bigger than 0, resp smaller than C
    alpha[alpha < 0] = 0
    alpha[alpha > C] = C

    return alpha

def selection(population, fits, k=3):
    # tournament selection
    idx = np.random.randint(0, len(population), k)
    return population[idx[np.argmax(fits[idx])]].copy()

def evolutive_algorithm(Q, y_train):
    #number of genes
    nr_alpha=Q.shape[0]

    #number of individuals from population
    population_size = 50

    #upper limit for alpha
    C=1.0

    # 1. Initialization
    population=np.random.uniform(0, C, (population_size,nr_alpha))

    # *test for adjustment function
    population=np.array([adjustment_function(ind, y_train, C) for ind in population])

    # 2. Fitness
    fits = np.array([fitness_function(ind, Q) for ind in population])

    print(f"Best individual: {np.max(fits)}")
    return population, fits



if __name__ == "__main__":
    x_train, x_test, y_train, y_test, Q = prepare_data()
    population, scores = evolutive_algorithm(Q, y_train)
