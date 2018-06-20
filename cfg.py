class sinkhorn:
    max_iter = 200
    lambda_ = 10
    float_tol = 1e-3

class train:
    save_freq = 10
    max_iter = 100
    batch_size = 32
    n_neighbors = 200
    lr_w = 1e+1
    lr_A = 1e+0
    w_min = 0.01
    w_max = 10
    use_baseline = False
    hidden_dim = 32

class data:
    datapath_train = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_train.csv'
    datapath_val = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_val.csv'
    datapath_test = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_test.csv'
    embeddings_path = '/Users/vlyalin/Downloads/lenta_lower_100.bin'
    savefolder = 'results'
    max_dict_size = 50000
