# Google Titans

## Some clarifications
In the paper, we basically have two kinds of `loops`:
  - Inner Loop: The inner loop is the memory updates that are custom defined for the MLP layers - based on the surprise, forget mechanism, step size, surprise factor
  - Outer Loop: This basically trains the hyperparameters used for training the inner loop [actual memory module]

## Problems
Neural memory implemented - only caveat is Im not sure if the w.data is correct or not for weight updation
