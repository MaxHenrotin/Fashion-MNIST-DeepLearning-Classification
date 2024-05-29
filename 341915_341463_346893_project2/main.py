import argparse
import random

import numpy as np
from torchinfo import summary
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    # xtrain = train_images (array): images of the train set, of shape (N,H,W) --> which will be reshaped to (N, V)
    # ytrain = train_labels (array): labels of the train set, of shape (N,)
    # xtest = test_images (array): images of the test set, of shape (N',H,W) --> which will be reshaped to (N', V)
    xtrain, xtest, ytrain = load_data(args.data) 
    xtrain = xtrain.reshape(xtrain.shape[0], -1)    #images are flatten to a vector
    xtest = xtest.reshape(xtest.shape[0], -1)       #images are flatten to a vector
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test :
    ### WRITE YOUR CODE HERE

        # Split the data into training and validation sets
        split_ratio = 0.8  # 80% training, 20% validation #ARBITRARY

        #mélange les datas
        size = len(ytrain)  #(=len(xtrain))
        pattern = [i for i in range(size)]
        random.shuffle(pattern)

        xtrain2 = np.empty_like(xtrain)
        ytrain2 = np.empty_like(ytrain)

        for i, p in enumerate(pattern):
            xtrain2[p] = xtrain[i]
            ytrain2[p] = ytrain[i]

        #crée les validationSet
        xtest = xtrain2[int(len(xtrain) * split_ratio):]
        ytest = ytrain2[int(len(ytrain) * split_ratio):]    #on possède donc maintenant également un ytest pour faire nos tests grâce au validation set

        #crée les trainingSet
        xtrain = xtrain2[:int(len(xtrain) * split_ratio)]
        ytrain = ytrain2[:int(len(ytrain) * split_ratio)]



    

    ### WRITE YOUR CODE HERE to do any other data processing

    #normalisation  we discovered that it gives worst performances with normalization
    #mu_train = np.mean(xtrain,0,keepdims=True)
    #std_train = np.std(xtrain,0,keepdims=True)
    #xtrain = normalize_fn(xtrain, mu_train, std_train)
    #xtest = normalize_fn(xtest, mu_train, std_train)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        pca_obj = PCA(d=args.pca_d) #pca_d = 300 seems a good tradeof because we have exvar = 95%
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        #initialize principal components and print explained variance
        exvar = pca_obj.find_principal_components(xtrain)
        print(f'The total variance explained by the first {args.pca_d} principal components is {exvar:.3f} %')
        #reduce dimensions
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        #should obviously not do pca on ytrain/ytest (because it's only a label)

    #biais appending not necessary because pytorch does it itself
    #xtrain = append_bias_term(xtrain)  #not necessary
    #xtest = append_bias_term(xtest)    #not necessary

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    input_channels = 1  # grayscale images
    height, width = 28, 28  # MNIST Image dimensions before flattening  # (= int(np.sqrt(xtrain.shape[1])))
    n_classes = get_n_classes(ytrain)   #number of classes/labels (= 10)


    if args.plotting : 
        print("Plotting")
        #lambdas = np.logspace(-3,1,num = 100,endpoint = True)
        iters = np.array([1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2])
        #iters = np.array([1e-7,5e-7,1e-6,5e-6])
        accuracy_1 = np.zeros(len(iters))

        xtrain = xtrain.reshape(-1, input_channels, height, width)  
        xtest = xtest.reshape(-1, input_channels, height, width)
        model = CNN(1, n_classes=n_classes)
        summary(model)


        for i in range(len(iters)):

            print("Lr : " + str(iters[i]))

            method_obj = Trainer(model, lr=iters[i], epochs=1, batch_size=args.nn_batch_size)
            preds_train = method_obj.fit(xtrain, ytrain)
            preds = method_obj.predict(xtest)
            accuracy_1[i] = accuracy_fn(preds, ytest)


        best_it = iters[np.argmax(accuracy_1)]
        best_it_formatted = format(best_it, ".0e")
        max_acc = np.max(accuracy_1)
        plt.scatter(best_it,max_acc,label = f"best learning rate : lr = {best_it_formatted}, accuracy = {max_acc:.3f}%", color = 'r')
        # Tracer les données des deux tableaux
        plt.plot(iters, accuracy_1, label='CNN',color = 'b')
        # Set x-axis to log scale
        plt.xscale('log')

        # Configure the x-axis to display in scientific notation
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0e}'))


        #plt.grid(True)
        # Ajouter des étiquettes d'axe et une légende
    
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Relation between learning rate and accuracy (max_iters = 5)')
        plt.legend()

        # Afficher le graphe
        plt.show()

    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
    if args.nn_type == "cnn":
        #reshape xtrain + xtest to size (N, 1, 28, 28)
        xtrain = xtrain.reshape(-1, input_channels, height, width)  
        xtest = xtest.reshape(-1, input_channels, height, width)
        model = CNN(input_channels= input_channels ,n_classes=n_classes)  #because we work with black and white images (only one channel and not 3)
    if args.nn_type == "transformer":
        #reshape xtrain + xtest to size (N, 1, 28, 28)
        xtrain = xtrain.reshape(-1, input_channels, height, width)
        xtest = xtest.reshape(-1, input_channels, height, width)
        n_patches = 7  #size of the patches that the image is divided into. each patch will be of size 4x4 (since 28/7=4). The number of patches in this case will be 49 (7x7). (7 is good for 28x28 images)
        n_blocks = 3    #determines the depth of the Transformer, i.e., how many layers of Transformer blocks are stacked. (2 is good to start but could be bigger)
        hidden_d = 64    #dimension of the hidden layers within the Transformer blocks (8 is good for 28x28 images)
        n_heads = 4     #number of heads in the multi-head attention mechanism (2 is often a good balance between performance and computational complexity)
        out_d = n_classes   
        model = MyViT(chw= (input_channels, height, width), n_patches= n_patches, n_blocks= n_blocks, hidden_d= hidden_d, n_heads= n_heads, out_d= out_d)

    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    
    parser.add_argument('--plotting',action="store_true", help="Executing the plot")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)