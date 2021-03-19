[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135605-ba0e5f2c-7d12-11e8-9578-86d74e0976f8.gif "Trained Agent"

# Cross-Entropy Method

### Instructions

Open `CEM.ipynb` to see an implementation of the cross-entropy method with OpenAI Gym's MountainCarContinuous environment.

Try to change the parameters in the notebook, to see if you can get the agent to train faster!

### Results

![Trained Agent][image1]

### Algorithm
```
func CEM():
  for i_iteration in Episodes:
    1. add noise to the last weights, and generate pop_size* randomed_weight
    2. run and estimate all pop_size sets of RETURN
    3. Select the ten sets of weights with the largest RETURN, and mean them get a best weights
    4. evaluate the RETURN with the new generated weights, append it in scores_deque
      break if mean of score_deque > 90

```
