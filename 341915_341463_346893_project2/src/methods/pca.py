import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """

        #### WRITE YOUR CODE HERE!

        # Compute the mean of the data
        self.mean = np.mean(training_data, axis=0)
        # Center the data with the mean
        centered_data = training_data - self.mean
        
        # Create the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        
        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort the eigenvalues and corresponding eigenvectors in decreasing order
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        
        # Choose the top d eigenvalues and corresponding eigenvectors
        self.W = sorted_eigenvectors[:, :self.d]
        topD_eigVec = sorted_eigenvalues[:self.d]
        
        # Compute the explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance = np.sum(topD_eigVec)
        exvar = (explained_variance / total_variance) * 100
        
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """

        #### WRITE YOUR CODE HERE!

        #find principal components
        exvar = self.find_principal_components(data)

        #Center the data by subtracting the mean
        centered_data = data - self.mean
        
        #Project the data onto the top 'd' eigenvectors
        data_reduced = np.dot(centered_data, self.W)

        print(f'The total variance explained by the first {self.d} principal components is {exvar:.3f} %')

        return data_reduced
        

