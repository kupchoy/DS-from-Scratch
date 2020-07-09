from typing import List

inputs: List[List[float]] = [[1., 49, 4, 0], [1, 41, 9, 0], [1, 40, 8, 0], [1, 25, 6, 0], [1, 21, 1, 0], [1, 21, 0, 0],
                             [1, 19, 3, 0], [1, 19, 0, 0], [1, 18, 9, 0], [1, 18, 8, 0], [1, 16, 4, 0], [1, 15, 3, 0],
                             [1, 15, 0, 0], [1, 15, 2, 0], [1, 15, 7, 0], [1, 14, 0, 0], [1, 14, 1, 0], [1, 13, 1, 0],
                             [1, 13, 7, 0], [1, 13, 4, 0], [1, 13, 2, 0], [1, 12, 5, 0], [1, 12, 0, 0], [1, 11, 9, 0],
                             [1, 10, 9, 0], [1, 10, 1, 0], [1, 10, 1, 0], [1, 10, 7, 0], [1, 10, 9, 0], [1, 10, 1, 0],
                             [1, 10, 6, 0], [1, 10, 6, 0], [1, 10, 8, 0], [1, 10, 10, 0], [1, 10, 6, 0], [1, 10, 0, 0],
                             [1, 10, 5, 0], [1, 10, 3, 0], [1, 10, 4, 0], [1, 9, 9, 0], [1, 9, 9, 0], [1, 9, 0, 0],
                             [1, 9, 0, 0], [1, 9, 6, 0], [1, 9, 10, 0], [1, 9, 8, 0], [1, 9, 5, 0], [1, 9, 2, 0],
                             [1, 9, 9, 0], [1, 9, 10, 0], [1, 9, 7, 0], [1, 9, 2, 0], [1, 9, 0, 0], [1, 9, 4, 0],
                             [1, 9, 6, 0], [1, 9, 4, 0], [1, 9, 7, 0], [1, 8, 3, 0], [1, 8, 2, 0], [1, 8, 4, 0],
                             [1, 8, 9, 0], [1, 8, 2, 0], [1, 8, 3, 0], [1, 8, 5, 0], [1, 8, 8, 0], [1, 8, 0, 0],
                             [1, 8, 9, 0], [1, 8, 10, 0], [1, 8, 5, 0], [1, 8, 5, 0], [1, 7, 5, 0], [1, 7, 5, 0],
                             [1, 7, 0, 0], [1, 7, 2, 0], [1, 7, 8, 0], [1, 7, 10, 0], [1, 7, 5, 0], [1, 7, 3, 0],
                             [1, 7, 3, 0], [1, 7, 6, 0], [1, 7, 7, 0], [1, 7, 7, 0], [1, 7, 9, 0], [1, 7, 3, 0],
                             [1, 7, 8, 0], [1, 6, 4, 0], [1, 6, 6, 0], [1, 6, 4, 0], [1, 6, 9, 0], [1, 6, 0, 0],
                             [1, 6, 1, 0], [1, 6, 4, 0], [1, 6, 1, 0], [1, 6, 0, 0], [1, 6, 7, 0], [1, 6, 0, 0],
                             [1, 6, 8, 0], [1, 6, 4, 0], [1, 6, 2, 1], [1, 6, 1, 1], [1, 6, 3, 1], [1, 6, 6, 1],
                             [1, 6, 4, 1], [1, 6, 4, 1], [1, 6, 1, 1], [1, 6, 3, 1], [1, 6, 4, 1], [1, 5, 1, 1],
                             [1, 5, 9, 1], [1, 5, 4, 1], [1, 5, 6, 1], [1, 5, 4, 1], [1, 5, 4, 1], [1, 5, 10, 1],
                             [1, 5, 5, 1], [1, 5, 2, 1], [1, 5, 4, 1], [1, 5, 4, 1], [1, 5, 9, 1], [1, 5, 3, 1],
                             [1, 5, 10, 1], [1, 5, 2, 1], [1, 5, 2, 1], [1, 5, 9, 1], [1, 4, 8, 1], [1, 4, 6, 1],
                             [1, 4, 0, 1], [1, 4, 10, 1], [1, 4, 5, 1], [1, 4, 10, 1], [1, 4, 9, 1], [1, 4, 1, 1],
                             [1, 4, 4, 1], [1, 4, 4, 1], [1, 4, 0, 1], [1, 4, 3, 1], [1, 4, 1, 1], [1, 4, 3, 1],
                             [1, 4, 2, 1], [1, 4, 4, 1], [1, 4, 4, 1], [1, 4, 8, 1], [1, 4, 2, 1], [1, 4, 4, 1],
                             [1, 3, 2, 1], [1, 3, 6, 1], [1, 3, 4, 1], [1, 3, 7, 1], [1, 3, 4, 1], [1, 3, 1, 1],
                             [1, 3, 10, 1], [1, 3, 3, 1], [1, 3, 4, 1], [1, 3, 7, 1], [1, 3, 5, 1], [1, 3, 6, 1],
                             [1, 3, 1, 1], [1, 3, 6, 1], [1, 3, 10, 1], [1, 3, 2, 1], [1, 3, 4, 1], [1, 3, 2, 1],
                             [1, 3, 1, 1], [1, 3, 5, 1], [1, 2, 4, 1], [1, 2, 2, 1], [1, 2, 8, 1], [1, 2, 3, 1],
                             [1, 2, 1, 1], [1, 2, 9, 1], [1, 2, 10, 1], [1, 2, 9, 1], [1, 2, 4, 1], [1, 2, 5, 1],
                             [1, 2, 0, 1], [1, 2, 9, 1], [1, 2, 9, 1], [1, 2, 0, 1], [1, 2, 1, 1], [1, 2, 1, 1],
                             [1, 2, 4, 1], [1, 1, 0, 1], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 5, 1], [1, 1, 3, 1],
                             [1, 1, 10, 1], [1, 1, 6, 1], [1, 1, 0, 1], [1, 1, 8, 1], [1, 1, 6, 1], [1, 1, 4, 1],
                             [1, 1, 9, 1], [1, 1, 9, 1], [1, 1, 4, 1], [1, 1, 2, 1], [1, 1, 9, 1], [1, 1, 0, 1],
                             [1, 1, 8, 1], [1, 1, 6, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 5, 1]]

from scratch.linear_algebra import dot, Vector


def predict(x: Vector, beta: Vector) -> float:
    """assumes that the first element of x is 1"""
    return dot(x, beta)


[1,  # constant term
 49,  # number of friends
 4,  # work hours per day
 0]  # doesn't have PhD

from typing import List


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


# %%


x = [1, 2, 3]
y = 30
beta = [4, 4, 4]

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36


# %%


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


assert sqerror_gradient(x, y, beta) == [-12, -24, -36]
# %%
import random
import tqdm
from scratch.linear_algebra import vector_mean
from scratch.gradient_descent import gradient_step


def least_squares_fit(xs: List[Vector],
                      ys: List[Vector],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta)
    """
    # Start with random guess
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y, in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


# %%
from scratch.statistics import daily_minutes_good
from scratch.gradient_descent import gradient_step

random.seed(0)
# I used trial and error to choose niters and step_size.
# This will run for a while.
learning_rate = 0.001

beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
assert 30.50 < beta[0] < 30.70  # constant
assert 0.96 < beta[1] < 1.00  # num friends
assert -1.89 < beta[2] < -1.85  # work hours per day
assert 0.91 < beta[3] < 0.94  # has PhD
# %%
from sklearn import linear_model

# Just wanted to see how it's in in sklearn and get same as in book
ols = linear_model.LinearRegression()
model = ols.fit(inputs, daily_minutes_good)

print(model.coef_)  # [ 0.          0.97250518 -1.86503639  0.9232007 ]
print(model.intercept_)  # 30.579018123991222
print(model.score(inputs, daily_minutes_good))  # 0.680011018137578 R-squared
# %%
from scratch.simple_linear_regression import total_sum_of_squares


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2
                                for x, y, in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


multiple_r_squared(inputs, daily_minutes_good, beta)  # 0.6799849346187969


# %%


def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


from scratch.linear_algebra import add


def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]


def sqerror_ridge_gradient(x: Vector,
                           y: float,
                           beta: Vector,
                           alpha: float) -> Vector:
    """
    the gradient corresponding to the ith squared error term
    including the ridge penalty
    """
    return add(sqerror_gradient(x, y, beta),
               ridge_penalty_gradient(beta, alpha))


learning_rate = 0.001


def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


# %%
random.seed(0)
beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0,  # alpha
                                 learning_rate, 5000, 25)
# [30.51, 0.97, -1.85, 0.91]
assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69

beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1,  # alpha
                                   learning_rate, 5000, 25)
# [30.8, 0.95, -1.83, 0.54]
assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1,  # alpha
                                 learning_rate, 5000, 25)
# [30.6, 0.90, -1.68, 0.10]
assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good, 10,  # alpha
                                  learning_rate, 5000, 25)
# [28.3, 0.67, -0.90, -0.01]
assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6
# %%
from sklearn import linear_model

# This has different solvers
rols = linear_model.Ridge()
rmodel = rols.fit(inputs, daily_minutes_good)

print(rmodel.coef_)  # [ 0.          0.97093209 -1.86416257  0.89373122]
print(rmodel.intercept_)  # 30.60105843663253
print(rmodel.score(inputs, daily_minutes_good))  # 0.6800095187110734
# %%
# alpha of 10 didn't change much.  Bumped it to 100 to get the PhD closer to 0
ridge_beta_100 = linear_model.Ridge(alpha=100,
                                    solver='lsqr',
                                    max_iter=5000).fit(inputs, daily_minutes_good)
print(ridge_beta_100.coef_)  # [ 0.          0.92400182 -1.77627852  0.19191323]
print(ridge_beta_100.intercept_)  # 30.88244684660524
print(ridge_beta_100.score(inputs, daily_minutes_good))  # 0.6782681853565671
# %%
# He mentioned lasso so I did this
lasso = linear_model.Lasso().fit(inputs, daily_minutes_good)
print(lasso.coef_)  # [ 0.          0.9005045  -1.76320824  0.        ]
print(lasso.intercept_)  # 31.08316362267097
print(lasso.score(inputs, daily_minutes_good))  # 0.6772804515825893

# %%
wtf = [1,  # constant term
       49,  # number of friends
       4,  # work hours per day
       1]
lass_wtf = lasso.predict([wtf])
print(lass_wtf)
ridge_wtf = rmodel.predict([wtf])
print(ridge_wtf)
