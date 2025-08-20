# Google Titans

## Some clarifications
In the paper, we basically have two kinds of `loops`:
  - Inner Loop: The inner loop is the memory updates that are custom defined for the MLP layers - based on the surprise, forget mechanism, step size, surprise factor
  - Outer Loop: This basically trains the hyperparameters used for training the inner loop [actual memory module]

## Problems
~~Neural memory implemented - only caveat is Im not sure if the w.data is correct or not for weight updation~~ -> fixed with naive update mechanism that just passes the weights back rather than changing them on the fly

## The idea behind Titans:
### Neural Memory
The principle idea is that when we learn, we learn more if something is more surprising to us. Take for instance, the idea of gravity, you know an apple falls and was the standard, but then the question of asking "why it falls" led us to think more deeply and learn from what might actually be happening. [Might be a bad example but I will hopefully come up with a better one **soon**]. But there are other things to consider as well, what if you were already surprised enough earlier by learning something similar and this new learning didn't attribute to your understanding that much? If the `momentary` surprise is huge that it is bigger than the previous surprise then we will learn to see something more deeper and learn something. [An example can be a reality check in an interview where your skills do not match your internal view of yourself and so you think "Ok, I need to learn more and my understanding has some connections missing" and so you learn something about yourself by that - which was the momentary surprise outweighing the previous]


Now, one thing that we also usually experience is forgetting irrelevant information or forgetting previously outdated ideas. The idea of flat earth was thrown off and forgotten, the moment circumnavigation was done [A Change in perspective and so forgetting was needed]. Another instance may be discarding the idea of "smells" transmitting diseases when Leeuwenhoek discovered micro-bacteria and organisms [new knowledge that required the old ideas to be forgotten]. Another aspect that helps is that, forgetting helps discard irrelevant information.
