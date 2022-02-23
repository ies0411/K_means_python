# Skeleton code for the task of implementing k-means
import numpy as np
from matplotlib import pyplot as plt
import copy
import datetime

# Use the set of pre-defined points given to you
import pickle
import sys

#this list valuable is for visualization of history
center_logs=[]
label_logs=[]

def distance(a, b):
    return (sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))])) ** 0.5

############### Your Helper Functions ###############################
def EStep(k_means, X):
  """
  The implementation of the expectation step of k-means
  """
  print("You need to implement the expectation step.")
  K = k_means.shape[0] # The number of clusters or the k of k-means
  D = k_means.shape[1] # The size of the feature vector
  T = X.shape[0] # The number of samples in X where X is of size (1000,2)

  # Track the distances and put them in a matrix
  res_X = np.zeros((T, K)) # there is one distance for each of the k means

  for k in range(K):
    print(k)
    # Do what you need to in order to calculate the distance for the samples to the current mean k
    for i in range(T):
      res_X[i,k]=distance(X[i],k_means[k,:])

  assignment_labels = np.zeros((T,)) # 1 label for each sample
  # You should have kept track of which cluster center was the best match,
  for i in range(T):
    assignment_labels[i] = np.argmin(res_X[i])

  # return this result so that you can use it in the next step
  return assignment_labels

def MStep(k_means, X, assignment_labels):
  """
  The implementation of the maximisation step of k-means
  """
  print("You need to implement the maximisation step.")
  # Update the cluster centers
  K = k_means.shape[0] # The number of clusters or the k of k-means
  D = k_means.shape[1] # The size of the feature vector
  T = X.shape[0] # The number of samples in X where X is of size (1000,2)

  # Go through and update each cluster center
  for k in range(K):
    print(k)

    # work out which samples belong to this cluster using the assignment labels
    points = [ X[j] for j in range(T) if assignment_labels[j] == k]
    # Update the cluster center using the equation for the maximisation step
    # print(points)
    k_means[k] = np.mean(points,axis=0)

  # Return the UPDATED cluster centers
  return k_means

def main():
  #### Load the data that you need to use
  with open('two_cluster_example.pickle','rb') as fp:
    (X1,X2) = pickle.load(fp, encoding='bytes')
    X = np.vstack((X1, X2)) # A matrix of size 1000x2
  # print(X)
  origin_x = X[:,0]
  origin_y = X[:,1]
  durations=[]
  #### Set the initial guess at to what the means should be
  k_means  = np.array([[-2.,1.],[0.,-1]]) # An array of size (2,2)
  # Write out the current cluster centers
  for k in range(k_means.shape[0]):
    print("For cluster",k, ": ", k_means[k,:])

  ### I'll get you to do this for 5 iterations, you can do more if you want

  for i in range(5):
    begin = datetime.datetime.now()
    # Get the labels of which cluster is response for which sample in X
    assignment_labels = EStep(k_means, X)

    # Update the cluster centers using the assignment of samples from the E-Step
    k_means = MStep(k_means, X, assignment_labels)
    end = datetime.datetime.now()
    duration = end-begin
    durations.append(duration.total_seconds())
    # Output the updated cluster centers for the current iteration
    print("At the end of iteration ", i+1, " we have the following update means.")
    for k in range(k_means.shape[0]):
      print("For cluster",k, ": ", k_means[k,:])
    tmp_k_means = copy.deepcopy(k_means)
    tmp_assignment_labels = copy.deepcopy(assignment_labels)

    center_logs.append(tmp_k_means)
    label_logs.append(tmp_assignment_labels)
  # visualization of k-means history
  fig, axes = plt.subplots(2,3,figsize=(12,7))
  ax = axes.flatten()
  plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.3,hspace=0.5)

  ax[0].scatter(X[:,0],X[:,1],c=label_logs[0],alpha=0.5)
  ax[0].scatter([-2.,1.],[0.,-1], c='red',label = 'centroid')
  ax[0].set_title("origin data distribution",fontsize=8)
  ax[0].legend()

  for i in range(1,6):
    if(i==5):
      ax[i].scatter(X[:,0],X[:,1],c=label_logs[i-1],alpha=0.5)
    else:
      ax[i].scatter(X[:,0],X[:,1],c=label_logs[i],alpha=0.5)

    ax[i].scatter(center_logs[i-1][:,0],center_logs[i-1][:,1], c='red',label = 'centroid')
    ax[i].set_title(f"{i} iteration ",fontsize=8)
    ax[i].text(-0.8,-2.1,"duration : "+ str(durations[i-1]) + "s")
    ax[i].legend()

  plt.show()



if __name__ == '__main__':
  main()

