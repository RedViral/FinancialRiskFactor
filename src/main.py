# import the data loader
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# import neural network
import network2

# import third party library for graphing
import matplotlib.pyplot as plt


# create a list to represent 3-layer network
    #  51 neurons in the input layer, corresponding to each data field per training example on the csv
    #  7 hidden neurons (random choice, may be changed)
    #  5 output neurons, corresponding to each of the 5 grade values (1-5)
list = [51, 7, 5]


# create a Network object
    # initilize object with a list (see above) 
    # list specifies network structure    
net = network2.Network(list)

# begin training your Network object
print "\nTraining..."

    # training takes place at:
    #     300 complete epochs
    #     275 training examples at one time (mini_batch_size)
    #     0.009 learning rate
    #     0.3 lmbda
    #     method params - (self, training_data, epochs, mini_batch_size, eta,lmbda = 0.0,
    #        evaluation_data=None, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False,
    #        monitor_training_cost=False, monitor_training_accuracy=False)

# For network2.py

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, 300, 75, 0.009,
    lmbda=0.3, 
    evaluation_data=test_data,
    monitor_evaluation_cost=True, 
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True, 
    monitor_training_accuracy=True,
    early_stopping_n = 100)

print "Training complete.\n"


# Additional graphs as performace analysis
# Figure 1 Shows the Cost value change over time for evalutation data
plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Evaluation Cost per Epoch')
plt.plot(evaluation_cost, color='r', linewidth=1.5)
plt.grid(True)

# Figure 2 shows the total accuracy change over time for evalutation data
plt.figure(2)
plt.xlabel('Epochs')
plt.ylabel('Correct Predictions')
plt.title('Evaluation Accuracy per Epoch')
plt.plot(evaluation_accuracy, color='b', linewidth=1.5)
plt.grid(True)

# Figure 3 shows accurate predictions per letter grade for evalutation data
# black shows As, blue Bs, green Cs, red Ds, cyan Fs
net.graphLetterGradeAccuracyResults()

# Show graphs
plt.show()
