---
layout: post
title: "First Test Post"
date: 2023-09-12 15:13:18 +0200
image: 12.jpg
tags: [jekyll, docs]
categories: jekyll
---

Roll a die with 20 sides.

The number rolled determines which secret cocktail you'll receive. [...]

You may re-roll if you land on something you rolled before, but rolling [any repeat] 3 times in a row means you get that number regardless. [...]

<img src="https://images.squarespace-cdn.com/content/v1/623506f9b920e800d3ca41c2/85dcaaea-ecd0-4bcd-a38c-2ea70a61e290/DM.png?format=2500w" width="300"/>

\begin{equation}
\sum_{i=1}^{20}\frac{1}{1-(1-(\frac{(20-i)}{20})^3)}
\end{equation}

``` python
# Change equation to eq^-1 instead of 1/eq

# Theoretical
def prob_of_new_number(n_already_rolled):
    return 1 - (1 - ((20 - n_already_rolled)/20))**3

expected_trials = []

for i in range(20):
    expected_trials.append(1/prob_of_new_number(i))

print('Expected number of trials: {}'.format(sum(expected_trials)))
```

    Expected number of trials: 33.860100253714165

``` python
# Simulation
import random
import math
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)

outcomes = []

def run_sim():
    numbers_already_rolled = []
    tries_left = 3
    number_of_trials = 0
    
    while len(numbers_already_rolled) < 20:
        current_roll = random.randint(1,20)
        if current_roll not in numbers_already_rolled:
            numbers_already_rolled.append(current_roll)
            tries_left = 3
            number_of_trials += 1
        elif tries_left > 1:
            tries_left -= 1
        else:
            tries_left = 3
            number_of_trials += 1
    
    return number_of_trials

for _ in range(500000):
    outcomes.append(run_sim())
    
discrete_trials = math.ceil(np.mean(outcomes))
print('Mean expected trials: {}'.format(discrete_trials))
print('Total cost ($16 per drink, plus tip and tax): ${})'.format(discrete_trials * 16 * 1.26))

plt.hist(outcomes, bins=20)
plt.show()
```

    Mean expected trials: 34
    Total cost ($16 per drink, plus tip and tax): $685.44)

![png](../../../../images/Parla_Calc_4_1.png)

``` python
```
