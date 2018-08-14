"classifying MNIST using k-means clustering"

import numpy as np
import matplotlib.pyplot as plt

def kmeans(k, dataset, epsilon=0):
    "calculate kmeans"
    history_centroids = []
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = np.linalg.norm(prototypes - prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = np.linalg.norm(prototypes - prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = np.linalg.norm(prototype - instance)

            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)
        #print(norm)

    return prototypes, history_centroids, belongs_to

if __name__ == "__main__":
    TRAINING_DATA = np.load('mnist.npz')['x_train'].reshape((-1, 784))
    for i in kmeans(10, TRAINING_DATA, 1)[0]:
    #for i in kmeans(100, TRAINING_DATA, 1)[0]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(i.reshape((28, 28)))
    plt.show()
