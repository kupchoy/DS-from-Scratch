def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha when actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i


from scratch.linear_algebra import Vector


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

#%%


from typing import Tuple
from scratch.statistics import correlation, standard_deviation, mean


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


#%%

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

# Should find that y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)

#%%

from scratch.statistics import num_friends_good, daily_minutes_good

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
#%%
from matplotlib import pyplot as plt

plt.scatter(num_friends_good, daily_minutes_good)
friends = [x for x in range(0, 50)]
minutes = [beta * friend + alpha for friend in friends]
plt.plot(friends, minutes)
plt.title("Simple Linear Regression Model")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()
#%%
from scratch.statistics import de_mean


def total_sum_of_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum( v**2 for v in de_mean(y))


def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
       the fraction of variation in y captured by the model, which equals
       1 - the fraction of variation in y not captured by the model
       """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))


rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)  # 0.33
#%%
print(rsq)
print(sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good))
print(total_sum_of_squares(daily_minutes_good))

#%%
import random
import tqdm
from scratch.gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        # Partial d of loss wrt alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i) for x_i, y_i in
                     zip(num_friends_good, daily_minutes_good))

        # Partial d of loss wrt beta
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good,
                                         daily_minutes_good))

        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")

        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

alpha, beta = guess
#%%
