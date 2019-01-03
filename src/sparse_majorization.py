import gensim
import sys
import numpy as np
import random

#Running sparse coding using majorization


PATH_TO_PRETRAINED_MODEL = "/Users/u6042446/Downloads/GoogleNews-vectors-negative300.bin"

#List of parameters
dim_embed = 300
dim_atoms = 600
marginal_rate = .1
num_internal_iterations = 100
landa = .005

size_of_mini_batch = 512


def landweber_shrinkage(A,landa):
    shrinkage = (A-(landa/2)*np.sign(A))*(np.abs(A)>(landa/2))
    return (shrinkage)


def initialize_dictionary(dim_embed,dim_atoms):
    temp_D = np.random.rand(dim_embed,dim_atoms)
    column_norms = np.linalg.norm(temp_D,axis=0).reshape([1,dim_atoms])
    temp_denum_matrix = np.ones([dim_embed,1]).dot(column_norms)
    return (temp_D/temp_denum_matrix)
    #normalize columns of dictionary

def initialize_sparse_matrix(dim_atoms,num_samples,num_non_zero_entries):
    init_sparse_coeff = np.zeros([dim_atoms,num_samples])
    for n in range(0,num_samples):
        #generate locations
        non_zero_locations = random.sample(range(0,dim_atoms),num_non_zero_entries)
        #generate random values
        vals = np.random.multivariate_normal(np.zeros(num_non_zero_entries), np.eye(num_non_zero_entries))
        init_sparse_coeff[:,n][non_zero_locations] = vals
    return (init_sparse_coeff)

def initialize_block_covs(dim_atoms,dim_embed,initial_tempreture,init_dict):
    block_sparse_cov = initial_tempreture*np.eye(dim_atoms)
    block_cross_cov = initial_tempreture*init_dict

    return (block_sparse_cov,block_cross_cov)





def sparse_vector_majorization_optimizer(init_sparse_matrix,_dictionary,dense_vectors,landa,block_sparse_cov,block_sparse_cross_cov,block_index):
    """
    Sparse matrix calculation using Majorization
    :param old_sparse_matrix: dim_atoms * mini_batch_size matrix of sparse representation
    :param _dictionary: dim_embed * dim_atoms dictionary matrix
    :param dense_vectors: dim_embed * mini_batch_size matrix of dense vectors
    :param landa:
    :return:
    """
    #choose second-order term
    sparse_matrix_keeper = [np.array(init_sparse_matrix)]
    #init_sparse_matrix = np.array(old_sparse_matrix)
    second_matrix_d = np.transpose(_dictionary).dot(_dictionary)
    cx = (1 + marginal_rate) * np.linalg.norm(second_matrix_d)
    eye_matrix = np.eye(dim_atoms)

    for n in range(0,num_internal_iterations):
        #Form Landweber Update
        A = (1/cx)*(np.transpose(_dictionary).dot(dense_vectors)+(cx*eye_matrix-second_matrix_d).dot(sparse_matrix_keeper[-1]))
        sparse_matrix_keeper.append(landweber_shrinkage(A,landa))

    #Coefficients from the reference (Online Dictionary Learning for Sparse coding)
    if block_index<size_of_mini_batch:
        theta = block_index*size_of_mini_batch
    else:
        theta = size_of_mini_batch**2+block_index-size_of_mini_batch

    beta = (theta + 1 - size_of_mini_batch) / (theta + 1)

    out_sparse_matrix = sparse_matrix_keeper[-1]
    #Update covariance and cross-covariance matrices
    block_sparse_cov = beta*block_sparse_cov+out_sparse_matrix.dot(out_sparse_matrix.transpose()) #A_t
    block_sparse_cross_cov = beta*block_sparse_cross_cov + dense_vectors.dot(out_sparse_matrix.transpose()) #B_t

    return(out_sparse_matrix,block_sparse_cov,block_sparse_cross_cov)

def mini_batch_dictionary_update(_dictionary,block_sparse_cov,block_sparse_cross_cov):

    """
    Runs dictionary refinement and outputs the updated dictionary
    :param _dictionary: the previous dictionary
    :param block_sparse_cov: A_t (covariance of sparse vectors)
    :param block_sparse_cross_cov: B_t (cross-covariance between dense and sparse vectors)
    :return:
    """

    num_iterations = 5
    (row_dim, col_dim) = np.shape(_dictionary)

    for n in range(0,num_iterations):
        #Update column by column
        for j in range(0,col_dim):
            temp_vec = (block_sparse_cross_cov[:,j]-_dictionary.dot(block_sparse_cov[:,j]))/(block_sparse_cov[j,j])+_dictionary[:,j]
            temp_vec = temp_vec/(max(1,(np.linalg.norm(temp_vec))**2))
            _dictionary[:,j] = temp_vec

    return (_dictionary)


def choose_mini_batch(dense_matrix,indexes,batch_size):
    if len(indexes)<=batch_size:
        return (dense_matrix[indexes,:],[])
    else:
        randomly_chosen_indexes = random.sample(indexes,batch_size)
        new_samples = indexes - set(randomly_chosen_indexes)
        return (dense_matrix[randomly_chosen_indexes,:],new_samples)


#def dictionary_refinement():

if __name__=="__main__":
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
    set_of_indexes = set([])
    dense_matrix = []
    for index,word in enumerate(model.index2word):
        dense_matrix.append(model.get_vector(word))
        set_of_indexes.add(index)
    dense_matrix = np.array(dense_matrix)

    #Mini-batch learning
    num_samples,dim_embed = np.shape(dense_matrix)
    all_indexes = set(range(0,num_samples))

    #Initialization
    dictionary = initialize_dictionary(dim_embed,dim_atoms)

    block_index = 0
    while not not len(all_indexes):
        print("processing block:"+str(block_index))
        mini_batch_dense_matrix,all_indexes = choose_mini_batch(dense_matrix,all_indexes,size_of_mini_batch)
        #initialize sparse-matrix
        sparse_matrix = initialize_sparse_matrix(dim_atoms,size_of_mini_batch,int(dim_atoms/10))
        sparse_block_cov,sparse_block_cross_cov = initialize_block_covs(dim_atoms,dim_embed,.1,dictionary)
        #sparcify
        sparse_matrix,sparse_block_cov,sparse_block_cross_cov = sparse_vector_majorization_optimizer(sparse_matrix,dictionary,mini_batch_dense_matrix,landa,sparse_block_cov,sparse_block_cross_cov,block_index)
        block_index+=1
        dictionary = mini_batch_dictionary_update(dictionary,sparse_block_cov,sparse_block_cross_cov)


    


