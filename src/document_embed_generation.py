


#base_path = "/Users/u6042446/Downloads/"
#base_path = "/data/sparse_coding/data/"
base_path = "/Users/u6042446/Desktop/backups/sparse_coding/data/glove/"
book_corpus_text = base_path+"all_tokanized_text.txt"
PATH_TO_PRETRAINED_WORDS = base_path+"glove.twitter.27B.50d.txt"

import linecache
from nltk.tokenize import word_tokenize
import random
import numpy as np
from sparse_lasso_glove import  *

n_epochs = 1
dim_embed = 50
dim_atoms = 200
marginal_rate = .1
num_internal_iterations = 100
landa = .00025
size_of_mini_batch = 512

def generate_mini_batch_of_sentences(line_numbers,embed_map):
    """
    Based on the line numbers of the original text-file, generat embedding matrix
    :param line_numbers:
    :param embed_map:
    :return:
    """
    dense_vector = []
    for line_number in line_numbers:
        sentence = linecache.getline(book_corpus_text,line_number).strip().split("\t")[-1]
        words = word_tokenize(sentence)
        embeds = []
        for x in words:
            if x in embed_map:
                embed = embed_map[x]
                embeds.append(embed)

        #generate average,max,min
        embeds = np.array(embeds)
        doc_embed = list(np.mean(embeds,axis=0))+list(np.max(embeds,axis=0))+list(np.min(embeds,axis=0))
        dense_vector.append(doc_embed)
    return (np.array(dense_vector))

def extract_embeds():
    embed_map = {}
    with open(PATH_TO_PRETRAINED_WORDS, "r") as f:
        for line in f:
            data = line.strip().split(" ")
            word = data[0]
            tmp_vec = data[1:]
            vec = []
            [vec.append(float(x)) for x in tmp_vec]
            embed_map[word] = vec

    return(embed_map)

if __name__=="__main__":


    #Metrics
    test_est_error = []
    test_sparsity = []
    test_total_error = []
    dict_update_error = []
    embed_map = extract_embeds()
    original_line_numbers = []
    with open(book_corpus_text,"r") as f:
        for index,line in enumerate(f):
            if line.strip()!=" " and line.strip()!="":
                original_line_numbers.append(index+1)

    # Initialize the dictionary
    dictionary = initialize_dictionary(dim_embed, dim_atoms)
    dictionary_saver = np.array(dictionary)
    sparse_block_cov, sparse_block_cross_cov = initialize_block_covs(dim_atoms, dim_embed, .1, dictionary)

    for epoch_index in range(0,10):
        print("Processing epoch:"+str(epoch_index))
        random.shuffle(original_line_numbers)
        num_samples_per_epoch = int(len(original_line_numbers) / 10)
        #Suffle all the lines
        #For each epoch: choose 10% of data for training
        line_numbers = original_line_numbers[0:num_samples_per_epoch]
        #random.shuffle(line_numbers)
        test_line_numbers, train_line_numbers = generate_train_test_indexes(line_numbers, .01)
        print("Total number of training examples:"+str(len(train_line_numbers)))
        #Extract test-matrix
        test_dense_matrix = generate_mini_batch_of_sentences(test_line_numbers,embed_map)
        #Number of blocks
        num_samples = len(train_line_numbers)
        num_blocks = num_samples / size_of_mini_batch

        block_index = 0
        while not not train_line_numbers:
            batch_line_numbers = train_line_numbers[0:size_of_mini_batch]
            train_line_numbers = train_line_numbers[size_of_mini_batch:]

            #stopping criterion
            if len(dict_update_error) > 0:
                if len(dict_update_error)>1:
                    rel_error = np.abs(dict_update_error[-1]-dict_update_error[-2])/dict_update_error[-2]
                else:
                    rel_error = 100
                if dict_update_error[-1]<.02 or rel_error<.01: break
            else:
                rel_error = 100

            if block_index % 10 == 0:
                total_error, est_error, sparsity = calculate_metrics(dictionary, np.transpose(test_dense_matrix))
                test_est_error.append(est_error)  # calculate test-error
                test_sparsity.append(sparsity)
                test_total_error.append(total_error)

            if block_index < size_of_mini_batch:
                theta = (block_index + 1) * size_of_mini_batch
            else:
                theta = size_of_mini_batch ** 2 + (block_index + 1) - size_of_mini_batch
            beta = (theta + 1 - size_of_mini_batch) / (theta + 1)

            mini_batch_dense_matrix = generate_mini_batch_of_sentences(train_line_numbers,embed_map) #training-batch
            (size_of_samples, z) = np.shape(mini_batch_dense_matrix)
            sparse_matrix = initialize_sparse_matrix(dim_atoms, size_of_samples, int(dim_atoms / 10))
            # sparcify
            sparse_matrix, sparse_block_cov, sparse_block_cross_cov, error = sparse_vector_lasso_optimizer(dictionary,
                                                                                                           np.transpose(
                                                                                                               mini_batch_dense_matrix),
                                                                                                           landa,
                                                                                                           sparse_block_cov,
                                                                                                           sparse_block_cross_cov,
                                                                                                           beta)


            #Dictionary update
            block_index += 1
            dictionary = mini_batch_dictionary_update(dictionary, sparse_block_cov, sparse_block_cross_cov)
            #Calculate errors
            dict_update_error.append(np.linalg.norm(dictionary - dictionary_saver))
            dictionary_saver = np.array(dictionary)
            if block_index % 10 == 0:
                init_index = max(0, block_index - 10)
                print(
                "(Total error,estimate_error,sparsity,dictionary error,relative dictionary error)=({0},{1},{2},{3},{4})".format(test_total_error[-1],
                                                                                                  test_est_error[-1],
                                                                                                  test_sparsity[-1],
                                                                                                  dict_update_error[-1],rel_error))

