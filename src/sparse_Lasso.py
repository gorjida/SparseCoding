import gensim
import sys
import numpy as np
import random
from sklearn import linear_model

# Running sparse coding using majorization


PATH_TO_PRETRAINED_MODEL = "/Users/u6042446/Downloads/GoogleNews-vectors-negative300.bin"

# List of parameters
n_epochs = 1
dim_embed = 300
dim_atoms = 600
marginal_rate = .1
num_internal_iterations = 100
landa = .0005
size_of_mini_batch = 1024


# Decaying updates
def gen_learning_rate(iteration, l_max, l_min, N_max):
    if iteration > N_max: return (l_min)
    alpha = 2 * l_max
    beta = np.log((alpha / l_min - 1)) / N_max
    return (alpha / (1 + np.exp(beta * iteration)))


def landweber_shrinkage(A, landa):
    shrinkage = (A - (landa / 2) * np.sign(A)) * (np.abs(A) > (landa / 2))
    return (shrinkage)


def initialize_dictionary(dim_embed, dim_atoms):
    temp_D = np.random.rand(dim_embed, dim_atoms)
    column_norms = np.linalg.norm(temp_D, axis=0).reshape([1, dim_atoms])
    temp_denum_matrix = np.ones([dim_embed, 1]).dot(column_norms)
    return (temp_D / temp_denum_matrix)
    # normalize columns of dictionary


def initialize_sparse_matrix(dim_atoms, num_samples, num_non_zero_entries):
    init_sparse_coeff = np.zeros([dim_atoms, num_samples])
    for n in range(0, num_samples):
        # generate locations
        non_zero_locations = random.sample(range(0, dim_atoms), num_non_zero_entries)
        # generate random values
        vals = np.random.multivariate_normal(np.zeros(num_non_zero_entries), np.eye(num_non_zero_entries))
        init_sparse_coeff[:, n][non_zero_locations] = vals
    return (init_sparse_coeff)


def initialize_block_covs(dim_atoms, dim_embed, initial_tempreture, init_dict):
    block_sparse_cov = initial_tempreture * np.eye(dim_atoms)
    block_cross_cov = initial_tempreture * init_dict

    return (block_sparse_cov, block_cross_cov)


def run_majorization(init_sparse_matrix, _dictionary, dense_vectors):
    # choose second-order term
    sparse_matrix_keeper = [np.array(init_sparse_matrix)]
    # init_sparse_matrix = np.array(old_sparse_matrix)
    second_matrix_d = np.transpose(_dictionary).dot(_dictionary)
    cx = (1 + marginal_rate) * np.linalg.norm(second_matrix_d)
    eye_matrix = np.eye(dim_atoms)

    for n in range(0, num_internal_iterations):
        # Form Landweber Update
        A = (1 / cx) * (np.transpose(_dictionary).dot(dense_vectors) + (cx * eye_matrix - second_matrix_d).dot(
            sparse_matrix_keeper[-1]))
        sparse_matrix_keeper.append(landweber_shrinkage(A, landa))

    return (sparse_matrix_keeper[-1])

def sparse_vector_lasso_optimizer(_dictionary,dense_vectors,landa,block_sparse_cov,
                                         block_sparse_cross_cov, beta):

    #Create LASSO model
    clf = linear_model.Lasso(alpha=landa)
    clf.fit(_dictionary,dense_vectors)
    #calculate error
    weights = clf.coef_
    W = np.transpose(weights)
    predicts = _dictionary.dot(np.transpose(weights))
    error = np.mean(np.linalg.norm(predicts-dense_vectors,axis=0))+landa*np.mean(np.linalg.norm(weights,ord=1,axis=1))

    block_sparse_cov = beta * block_sparse_cov + W.dot(W.transpose())  # A_t
    block_sparse_cross_cov = beta * block_sparse_cross_cov + dense_vectors.dot(W.transpose())  # B_t

    return (W,block_sparse_cov, block_sparse_cross_cov,error)


def mini_batch_dictionary_update(_dictionary, block_sparse_cov, block_sparse_cross_cov):
    """
    Runs dictionary refinement and outputs the updated dictionary
    :param _dictionary: the previous dictionary
    :param block_sparse_cov: A_t (covariance of sparse vectors)
    :param block_sparse_cross_cov: B_t (cross-covariance between dense and sparse vectors)
    :return:
    """

    num_iterations = 5
    (row_dim, col_dim) = np.shape(_dictionary)

    for n in range(0, num_iterations):
        # Update column by column
        for j in range(0, col_dim):
            temp_vec = (block_sparse_cross_cov[:, j] - _dictionary.dot(block_sparse_cov[:, j])) / (
            block_sparse_cov[j, j]) + _dictionary[:, j]
            temp_vec = temp_vec / (max(1, (np.linalg.norm(temp_vec)) ** 2))
            _dictionary[:, j] = temp_vec

    return (_dictionary)


def choose_mini_batch(dense_matrix, indexes, batch_size):
    if len(indexes) <= batch_size:
        return (dense_matrix[list(indexes), :], [])
    else:
        randomly_chosen_indexes = random.sample(indexes, batch_size)
        new_samples = indexes - set(randomly_chosen_indexes)
        return (dense_matrix[randomly_chosen_indexes, :], new_samples)


def calculate_metrics(dictionary, test_dense_matirx):
    dim_embed, num_samples = np.shape(test_dense_matirx)
    sparse_matrix = initialize_sparse_matrix(dim_atoms, num_samples, int(dim_atoms / 10))
    sparse_matrix = run_majorization(sparse_matrix, dictionary, test_dense_matirx)
    # Calculate the objective
    output = dictionary.dot(sparse_matrix)  # this is the predicted output
    avg_error = np.mean(np.linalg.norm(test_dense_matirx - output, axis=0))
    # calculate impact of sparsity
    sparsity_impact = landa * np.mean(np.linalg.norm(sparse_matrix, ord=1, axis=0))
    # print(sparsity_impact)
    return (avg_error + sparsity_impact)


def generate_train_test_indexes(all_indices, test_portion):
    """
    :Generating all the train/test indices
    :param all_indices:
    :param test_portion:
    :return:
    """
    random.shuffle(all_indices)
    test_num = int(len(all_indices) * test_portion)
    return (all_indices[0:test_num], all_indices[test_num:])


# def dictionary_refinement():

if __name__ == "__main__":

    """

    all_words_embedding = {}  # embedding for each single word appearing in the query or document-set
    word_to_word_similarity = {}

    print("Loading pre-trained model...")
    model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_PRETRAINED_MODEL, binary=True,
                                                            unicode_errors='ignore')
    # Build inverse dictionary (real-word TO index)
    inv_dict = dict((v, k) for k, v in enumerate(model.index2word))
    # list_of_embeddings
    pretrained_embeddings = model.syn0

    #Form matrix of embedding (dense matrix)
    writer_sampled = open("/Users/u6042446/Desktop/ali_files/sparse_coding/data/sampled_embed.txt","w")
    writer_all = open("/Users/u6042446/Desktop/ali_files/sparse_coding/data/embeds.txt", "w")

    set_of_indexes = set([])
    dense_matrix = []
    for index,word in enumerate(model.index2word):
        vec = []
        [vec.append(str(x)) for x in model.get_vector(word)]
        dense_matrix.append(model.get_vector(word))
        set_of_indexes.add(index)
        writer_all.write(word+"\t"+"|".join(vec)+"\n")
        if random.random()<.05: writer_sampled.write(word+"\t"+"|".join(vec)+"\n")

    writer_sampled.close()
    writer_all.close()
    sys.exit(1)
    """

    dense_matrix = []
    set_of_indexes = set([])
    with open("/Users/u6042446/Desktop/ali_files/sparse_coding/data/sampled_embed.txt", "r") as f:
        for index, line in enumerate(f):
            data = line.strip().split("\t")
            vec = []
            splited_data = data[1].split("|")
            [vec.append(float(x)) for x in splited_data]
            dense_matrix.append(vec)
            set_of_indexes.add(index)

    dense_matrix = np.array(dense_matrix)
    # Initialization
    dictionary = initialize_dictionary(dim_embed, dim_atoms)

    dictionary_saver = np.array(dictionary)

    test_error = []
    dict_update_error = []

    block_index = 0
    beta = 0
    for epoch_index in range(0, n_epochs):
        print("Epoch:" + str(epoch_index))
        # Mini-batch learning
        # split to train and test
        test_index, train_index = generate_train_test_indexes(list(set_of_indexes), .1)
        train_dense_matrix = dense_matrix[train_index, :]
        test_dense_matrix = dense_matrix[test_index, :]

        num_samples, dim_embed = np.shape(train_dense_matrix)
        all_indexes = set(range(0, num_samples))
        num_blocks = num_samples / size_of_mini_batch  # Number of blocks
        sparse_block_cov, sparse_block_cross_cov = initialize_block_covs(dim_atoms, dim_embed, .1, dictionary)

        while not not len(all_indexes):
            if block_index%10==0:test_error.append(calculate_metrics(dictionary,np.transpose(test_dense_matrix))) #calculate test-error
            # if block_index % 5 == 0: beta = min(.99, beta + .05)
            # print(beta)

            if block_index < size_of_mini_batch:
                theta = (block_index+1) * size_of_mini_batch
            else:
                theta = size_of_mini_batch ** 2 + (block_index+1) - size_of_mini_batch
            beta = (theta + 1 - size_of_mini_batch) / (theta + 1)
            #beta = .1

            print("processing block:" + str(block_index))
            mini_batch_dense_matrix, all_indexes = choose_mini_batch(train_dense_matrix, all_indexes,
                                                                     size_of_mini_batch)


            # initialize sparse-matrix
            (size_of_samples, z) = np.shape(mini_batch_dense_matrix)
            sparse_matrix = initialize_sparse_matrix(dim_atoms, size_of_samples, int(dim_atoms / 10))

            # sparcify
            sparse_matrix, sparse_block_cov, sparse_block_cross_cov,error = sparse_vector_lasso_optimizer(dictionary,np.transpose(mini_batch_dense_matrix),landa, sparse_block_cov,sparse_block_cross_cov, beta)
            block_index += 1
            dictionary = mini_batch_dictionary_update(dictionary, sparse_block_cov, sparse_block_cross_cov)
            keep_prev_dictionary = np.array(dictionary_saver)
            dict_update_error.append(np.linalg.norm(dictionary - dictionary_saver))
            #print(dict_update_error)
            # print(dict_update_error)
            dictionary_saver = np.array(dictionary)
            if block_index % 10 == 0: print(dict_update_error[-1])

            # calculate the performance (error)


            # print(test_error)

            # sys.exit(1)







