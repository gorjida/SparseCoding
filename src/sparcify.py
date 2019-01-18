import gensim
import sys
import numpy as np
import random
from sklearn import linear_model


from gram_schmidt import gramschmidt

# Running sparse coding using majorization

base_path = "/Users/u6042446/Downloads/"
base_path = "/data/sparse_coding/data/"
PATH_TO_PRETRAINED_WORDS = base_path+"glove.twitter.27B.50d.txt"

# List of parameters
n_epochs = 1
dim_embed = 50
dim_atoms = 200
marginal_rate = .1
num_internal_iterations = 100
landa = .003
size_of_mini_batch = 512
import linecache


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


def lasso_sparcify(_dictionary,dense_vectors):
    # Create LASSO model
    clf = linear_model.Lasso(alpha=landa)
    clf.fit(_dictionary, dense_vectors)
    # calculate error
    weights = clf.coef_
    return (weights)

def sparse_vector_lasso_optimizer(_dictionary,dense_vectors,landa,block_sparse_cov,
                                         block_sparse_cross_cov, beta):
    weights = lasso_sparcify(_dictionary,dense_vectors)
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


def create_dense_matrix(model,indicies):
    dense_matrix = []
    for t in indicies: dense_matrix.append(model.get_vector(model.index2word[t]))
    dense_matrix = np.array(dense_matrix)
    return (dense_matrix)


def choose_mini_batch(indexes,batch_size):
    if len(indexes)<=batch_size:
        return (list(indexes),[])
    else:
        randomly_chosen_indexes = random.sample(indexes,batch_size)
        new_samples = indexes - set(randomly_chosen_indexes)
        return (list(randomly_chosen_indexes),new_samples)

def calculate_metrics(dictionary, test_dense_matrix):
    # Create LASSO model

    clf = linear_model.Lasso(alpha=landa,max_iter=1000)
    clf.fit(dictionary, test_dense_matrix)
    # calculate error
    weights = clf.coef_
    sparse_matrix = np.transpose(weights)
    #dim_embed, num_samples = np.shape(test_dense_matirx)
    # Calculate the objective
    output = dictionary.dot(sparse_matrix)  # this is the predicted output
    avg_error = np.mean(np.linalg.norm(test_dense_matrix - output, axis=0))
    # calculate impact of sparsity
    sparsity_impact = landa * np.mean(np.linalg.norm(sparse_matrix, ord=1, axis=0))
    sparsity = np.mean(np.sum(np.abs(np.sign(sparse_matrix)),axis=0))/dim_atoms
    # print(sparsity_impact)
    return (avg_error + sparsity_impact,avg_error,sparsity)


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

def convert_sparse_vec_to_string(vec):
    string = []
    for index,val in enumerate(vec):
        if np.abs(val)>1E-4:
            string.append(str(index)+":"+str(val))
    return (string)

def generate_dense_batch(line_numbers,index_to_word_map):

    dense_matrix = []
    for l in line_numbers:
        line_number = l+1
        line = linecache.getline(PATH_TO_PRETRAINED_WORDS,line_number)
        data = line.strip().split(" ")
        word = data[0]
        index_to_word_map[l] = word
        vector = data[1:]
        vec = []
        [vec.append(float(x)) for x in vector]
        dense_matrix.append(vec)
    return (np.array(dense_matrix))


# def dictionary_refinement():

if __name__ == "__main__":
    #writer_sampled.close()
    #writer_all.close()
    #sys.exit(1)

    import sys
    #args = sys.argv
    #dim_atoms = int(args[1])
    #landa = float(args[2])

    dim_atoms = 200
    landa = .003
    writer = open(base_path+"sparse_vectors_{0}_{1}_50.txt".format(dim_atoms,landa),"w")

    dictionary_file = "dictionary"+"_"+str(dim_atoms)+"_"+str(dim_embed)+"_"+str(landa)+".npy"
    dictionary = np.load(base_path + dictionary_file)
    print("Running with num_atoms="+str(dim_atoms)+" and landa="+str(landa))
    apply_orth = False

    index_to_word_map = {}
    dense_matrix = []
    set_of_indexes = set([])
    with open(PATH_TO_PRETRAINED_WORDS, "r") as f:
        for index, line in enumerate(f):
            #data = line.strip().split(" ")
            #word = data[0]
            #vector = data[1:]
            #vec = []
            #[vec.append(float(x)) for x in vector]
            #dense_matrix.append(vec)
            set_of_indexes.add(index)
            #index_to_word_map[index] = word

    #dense_matrix = np.array(dense_matrix)
    #(num_samples,dim_embed) = np.shape(dense_matrix)
    num_samples = len(set_of_indexes)
    num_blocks = num_samples / size_of_mini_batch
    all_indexes = set(range(0, num_samples))
    word_to_sparse = {}
    num_blocks = 0
    while not not len(all_indexes):
        if num_blocks%10==0: print("Number of processed blocks is:{0}".format(num_blocks))
        chosen_indexes,all_indexes = choose_mini_batch(dense_matrix, all_indexes, size_of_mini_batch)
        mini_batch_dense_matrix = generate_dense_batch(chosen_indexes,index_to_word_map)
        try:
            weights = lasso_sparcify(dictionary,mini_batch_dense_matrix.transpose())
        except:
            print("skipping batch...")
            continue
        sparse_matrix = weights.transpose()

        for counter,index in enumerate(chosen_indexes):
            word_to_sparse[index_to_word_map[index]] = sparse_matrix[:,counter]
            str_ = convert_sparse_vec_to_string(sparse_matrix[:,counter])
            writer.write(index_to_word_map[index]+"\t"+"\t".join(str_)+"\n")
        num_blocks+=1

    writer.close()