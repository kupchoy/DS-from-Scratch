def double(x):
    """
    This function multiplies its input by 2
    :param x:
    :return: 2*x
    """
    return x * 2

def apply_to_one(f):
    """Calls the function f with 1 as argument"""
    return f(1)
#%%
my_double = double
x = apply_to_one(my_double)

assert x == 2
#%%
y = apply_to_one(lambda x: x + 4)
#%%
def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last

full_name("Joel", "Grus")     # "Joel Grus"
full_name("Joel")             # "Joel Something"
full_name(last="Grus")        # "What's-his-name Grus"
#%%
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3
#%%
try:
    my_tuple[1] = 3
except TypeError:
    print('cannot modify a tuple')
#%%
def sum_and_product(x, y):
    return (x + y), (x * y)


sp = sum_and_product(2, 3)
s, p = sum_and_product(5, 10)
#%%
x, y = 1, 2
x, y = y, x
#%%

grades = {'Joel': 80,
          'Tim': 95}
joels_grade = grades['Joel']
#%%
joels_grade = grades.get("Joel", 0)   # equals 80
kates_grade = grades.get("Kate", 0)   # equals 0
no_ones_grade = grades.get("No One")  # default default is None
#%%
tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}
#%%
tweet_keys   = tweet.keys()     # iterable for the keys
tweet_values = tweet.values()   # iterable for the values
tweet_items  = tweet.items()    # iterable for the (key, value) tuples
#%%
"user" in tweet_keys            # True, but not Pythonic
"user" in tweet                 # Pythonic way of checking for keys
"joelgrus" in tweet_values      # True (slow but the only way to check)
#%%
document = ["data", "science", "from", "scratch"]

word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1
#%%
from collections import defaultdict

word_counts = defaultdict(int)
#%%
for word in document:
    word_counts[word] += 1
#%%
from collections import Counter
c = Counter([0, 1, 2, 0])
#%%
word_counts = Counter(document)