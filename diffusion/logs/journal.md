# Experiments log

## Exp 1
Used the cyclic data to create `num_frames` variations of the same sequence
Just tried setting up the model and got training and sampling to work
Trained with 5000 steps, loss was around 1300
Results were bad, sequence was not maintained, and the model was not able to learn the distribution of the data

## Exp 2: More epochs
Tried training with more training steps and epochs => 20000 steps
Loss was still the same
Results were still bad

## Exp 3: Same train data
Suspect that the model learns from a fixed sequence and not from its variations
Took the sequence and multiplied it by `num_frames` to get the same training data size as before
Trained with 20000 steps
Loss was still around 1300
Results looked better but were still bad

## Exp 4: More train data
Used the same sequence but multiplied it by 100 this time, resulting in 2900 training samples
Trained with 20000 steps
Loss was still around 1000
Results looked better but order of sequence was still not maintained

## Exp 5: More mixed data
Repeated exp 4 but with multiples of variations of the same sequence
Results looked worse

## Exp 6: More training dims
Realized the previous attempts I was only using 32 dimensions for embeddings, bumped it up to 64
Repeated exp 5
Results looked better but still not good

### Thoughts
- Not using enough dimensions
  - using 32, MDM uses 512
- Data not in the right format
  - pos array + vel array makes no sense, position encoding will mess this up
  - MDM does first pose + vel array so sequence is conserved
- Loss is not optimized
  - Model just aims to predict pos array + vel array using pure geometric MSE
      - MDM first pose + vel array as the base loss, then has another process to calculate the pos array + vel array + foot contact and do geometric loss from those 
      - Uneven joint config and velocity => pos has 35 and vel has 34

## Exp 7: Model v2
Realized that order might be bad because I am just concatenating the pos and the vel arrays
Position encoding will mess this up, decide to follow MDM to use first pose + array of velocities
Don't know the format of the frame data
Mess around with input data 