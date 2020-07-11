from scipy.spatial.distance import hamming, cosine, euclidean

def cosine_distance(training_set_vectors, query_vector, top_n=30):
    
    distances = []
    # comparing each image to all training set
    for i in range(len(training_set_vectors)):
        distances.append(cosine(training_set_vectors[i], query_vector[0]))
    # return sorted indices of 30 most similar images
    return np.argsort(distances)[:top_n]

def hamming_distance(training_set_vectors, query_vector, top_n=50):

    distances = []
    # comparing each image to all training set
    for i in range(len(training_set_vectors)):
        distances.append(hamming(training_set_vectors[i], query_vector[0]))
    # return sorted indices of 30 most similar images   
    return np.argsort(distances)[:top_n]

def sparse_accuracy(true_labels, predicted_labels):

    # np array real labels of each sample
    # np matrix softmax probabilities
    
    assert len(true_labels) == len(predicted_labels)
    
    correct = 0
    for i in range(len(true_labels)):
        if np.argmax(predicted_labels[i]) == true_labels[i]:
            correct += 1
            
    return correct / len(true_labels)

def compare_color(color_vectors,         # color features vectors of closest training set images to the uploaded image            
                  uploaded_image_colors, # color vector of the uploaded image
                  ids):                  # indices of training images being closest to the uploaded image (output from a distance function)    

    color_distances = []
    
    for i in range(len(color_vectors)):
        color_distances.append(euclidean(color_vectors[i], uploaded_image_colors))
        
    # The 15 is just an random number that I have choosen, you can return as many as you need/want
    return ids[np.argsort(color_distances)[:15]]
