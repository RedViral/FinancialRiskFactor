Start of ReadMe

Dependencies:
To execute code you do not need a GPU, but please make sure you have your dependencies —
python, numpy and matplotlib.pyplot.

Execution:
Please make sure you are in the /src directory.

$ python main.py

Evaluation focus:
- Training the feedforward neural network with normalized data using financial letter grade as a bench mark (1 being best and 5 being worst). Execute training with randomized bias and weights to see optimal accuracy with provided parameters. 

Evaluation data results:
- Cost decreased with every epoch and were able to achieve up to 30% accuracy on the validation set. 

Data set details:
- Training Data: 1250 random rows (each letter grade is represented at 250)
- Test Data: 60 random rows
- Validation Data: was not used in this implementation

Structure of files and folders is:
/src/network2.py
/src/mnist_loader.py
/src/main.py
/data/(Your training data)
/data/(Your test data)
/data/(Your training data,please note that there is no validation data for this implementation.)

There is space for lot of correction in this file so please add or edit the structure for the best result. The code does need many improvements and posting back the changes whould help future problem solvers.
EOF
