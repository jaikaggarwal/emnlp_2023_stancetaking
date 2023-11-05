from core_utils import *

from functools import reduce
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords

ROOT_DIR = "INSERT HERE"
UNMASKED_ROOT_DIR = "INSERT HERE"

FEATURE_COLUMNS = ['Valence', 'Arousal', 'Dominance', 'Politeness', 'Formality']
NUM_QUANTILES = 3


# Maps each feature to its quantile number in the semantic situation name (e.g. V1A2D1P2F1)
feature_to_index = {
    "Valence": 1,
    "Arousal": 3,
    "Dominance": 5,
    "Politeness": 7,
    "Formality": 9
}

# Used for plotting MDS groups
quantile_to_plot_markers = {
    2: ["ro", "gs"],
    3: ["ro", "bx", "gs"],
    4: ["ro", "bo", "ys", "gs"]
}

# Used for labelling MDS groups
quantile_to_legend_labels = {
    2: ["Low", "High"],
    3: ["Low", "Mid", "High"],
    4: ["Low", "Midlow", "Midhigh", "High"]
}


def load_intensifiers():
    """
    Loads the set of intensifiers that are in our overall Reddit dataset. Removes any
    intensifiers that are also stopwords ("most", "very").
    """
    intensifiers = pd.read_csv("~/stancetaking/data_files/intensifiers.csv")['word'].unique()
    all_markers = Serialization.load_obj("pavalanathan_cooc_data_full_data").columns.tolist()
    valid_intensifiers = intersect_overlap(intensifiers, all_markers)
    english_stopwords = stopwords.words("english")
    valid_intensifiers = set_difference(valid_intensifiers, english_stopwords)
    return valid_intensifiers

    
def get_bins(data, num_bins, columns_of_interest):
    """Return the bin index that each data point in data falls into, given the space
    is subdivided to have num_bins equally sized bins.

    A bin number of i means that the corresponding value is between bin_edges[i-1], bin_edges[i]

    Returns both the bin index as a unique integer, as well as in terms of a 5d
    array corresponding to each dimension.
    """
    # Initialize uniformly-sized bins
    bin_edges = []
    for feature in columns_of_interest:
        bin_edges.append(np.quantile(data[feature], np.linspace(0, 1, num_bins + 1)))
    bin_edges = np.array(bin_edges)
    bin_edges[:, 0] = 0
    bin_edges[:, -1] = 1

    data = data.to_numpy()
    
    stats, edges, unraveled_binnumber = binned_statistic_dd(data, np.arange(len(data)),
                                                            statistic="mean",
                                                            bins=bin_edges,
                                                            expand_binnumbers=True)

    # Return the bin IDs
    return unraveled_binnumber.transpose()


def get_bin_names(arr, namespace):
    """
    Convert a bin's score on each dimension to the full bin name (along the lines of V1A1D2P3F4).

    Args:
        arr (np.array): scores of a given bin on each for each of the features
        namespace (str): string representation of the features we care about (e.g. VADPF)
    """
    features = np.array(list(namespace))
    added = np.char.add(features, arr.astype(str))
    names = np.sum(added.astype(object), axis=1)
    return names



def load_data_from_raw(save_suffix, markers_of_interest, namespace, columns_of_interest=FEATURE_COLUMNS):
    """
    Create the VADPF features scores for each sentence in our dataset, including only the markers of interest.
    """
    files = sorted(os.listdir(ROOT_DIR))
    print(len(files))
    print(files)
    dfs = []
    for file in tqdm(files):
        df = pd.read_csv(ROOT_DIR + file)
        dfs.append(df)
    df = pd.concat(dfs)
    del dfs
    df = df.set_index("id")
    df['rel_marker'] = df['rel_marker'].progress_apply(lambda x: x.strip("['']"))
    df = df[df['rel_marker'].isin(markers_of_interest)]

    print("Rescaling Politeness")
    df['Politeness'] = (df["Politeness"] - df['Politeness'].min())/(df['Politeness'].max() - df['Politeness'].min())


    print("Extracting Bins")
    ubins = get_bins(df[columns_of_interest], NUM_QUANTILES, columns_of_interest)
    print("Getting bin names")
    df['bin'] = get_bin_names(ubins, namespace)
    print("Describing bin")
    df['bin'].describe()

    #TODO: REMOVE OR REWRITE
    Serialization.save_obj(df, f"stance_pipeline_full_data_with_sentences_{save_suffix}")

    print("Getting mean data")
    x = df.groupby("bin").mean()[columns_of_interest]
    print("Saving mean data")
    Serialization.save_obj(x, f"semantic_situation_mean_values_{NUM_QUANTILES}_{save_suffix}")
    print("Saving entire dataset")
    Serialization.save_obj(df[['subreddit', 'rel_marker', 'bin'] + columns_of_interest], f"stance_pipeline_full_data_{NUM_QUANTILES}_quantiles_{save_suffix}")



def get_sub_marker_pairs(df):
    """
    Create all pairs of subreddits and markers and add to the dataframe. Filter out noisey markers like "'d" and "10x".
    """
    all_markers = sorted(df['rel_marker'].unique())
    all_markers = [marker for marker in all_markers if marker not in ["'d", "10x"]]
    df = df[df['rel_marker'].isin(all_markers)]
    # Combine the subreddit and marker and aggregate
    df['sub_marker'] = df["subreddit"] + "_" + df['rel_marker']
    return df


def get_bin_com_markers(df):
    """
    Return the unique communities, markers, bins, and community_marker pairs in the dataset.
    """
    comms = df['subreddit'].unique()
    markers = df['rel_marker'].unique()
    bins = df['bin'].unique()
    com_markers = list(product(comms, markers))
    com_markers = ["_".join(pair) for pair in com_markers]
    return bins, comms, markers, com_markers


def get_need_probabilities(df, bins, comms):
    """
    Compute the need probabilities each commmunity has for each of the semantic situations (bins). 
    Returns:
        com_to_df (dict): keys are the community names (lowercased) and values are the need probability vectors
        need_df (pd.Dataframe): dataframe version of com_to_df
    """
    # Need probability
    # Takes 30 seconds to run


    sem_sit_counts_per_community = df.groupby(["subreddit", "bin"]).count()[['sub_marker', "Valence"]]
    all_sub_counts = pd.DataFrame(0, index=pd.MultiIndex.from_product([bins, comms], names=["bin", "subreddit"]), columns=sem_sit_counts_per_community.columns)
    sem_sit_counts_per_community = sem_sit_counts_per_community.add(all_sub_counts, fill_value=0)
    sem_sit_counts_per_community['percent'] = sem_sit_counts_per_community.groupby(level=0)['sub_marker'].transform(lambda x: (x / x.sum()))

    com_to_need = {}
    for sub in comms:
        need_vec = sem_sit_counts_per_community.loc[sub]['percent']
        com_to_need[sub] = need_vec.to_numpy()
    
    need_df = pd.DataFrame(com_to_need).T
    need_df.columns = sem_sit_counts_per_community.loc[sub].index
    return com_to_need, need_df


def get_need_probabilities_wrapper(data_suffix):    
    df = Serialization.load_obj(f"stance_pipeline_full_data_{NUM_QUANTILES}_quantiles_{data_suffix}")

    all_markers = sorted(df['rel_marker'].unique())
    all_markers = [marker for marker in all_markers if marker not in ["'d", "10x"]]
    df = df[df['rel_marker'].isin(all_markers)]
    # Combine the subreddit and marker and aggregate
    df['sub_marker'] = df["subreddit"] + "_" + df['rel_marker']

    comms = df['subreddit'].unique()
    bins = df['bin'].unique()

    com_to_need, need_df = get_need_probabilities(df, bins, comms)
    return com_to_need, need_df


def get_nonzero_prop(df):
    """
    Return the number of non-zero elements in a dataframe, rounded to 2 digits.
    """
    print(np.round(np.count_nonzero(df)/df.size, 2))




def pmi(df, positive=True):
    """
    Compute the pointwise mutual information of each cell in a dataframe. The default is
    to return positive PMI.
    """
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
        df = np.nan_to_num(df)
    return df


def generate_latent_representations(matrix, label, title, data_suffix, user="jai", include_ppmi=True):
    """
    Applies SVD to the co-occurrence matrix. Optionally applies PMI metric.
    Note that loadings are computed as the eigenvectors multiplied by the square 
    root of the eigenvalues (which are the singular values).

    Returns a dictionary with:
        svd_input: the numpy form of the co-occurrence matrix that was passed in
        sem_rep: the latent representations for the semantic situations
        singular_values: the singular values returned by SVD
        marker_rep: the latent representations for the subreddit_marker pairs
        sem_loadings: the loadings of the semantic situations on each dimension
        marker_loadings: the loadings of the subreddit_marker pairs on each dimension
    """
    if not isinstance(matrix, np.ndarray):
        matrix_np = matrix.to_numpy()
        print(matrix_np.shape)
    else:
        matrix_np = matrix
        
    if include_ppmi:
        svd_input = pmi(matrix_np)
    else:
        svd_input = matrix_np

    get_nonzero_prop(svd_input)

    P, D, Q = np.linalg.svd(svd_input, full_matrices=False)
    
    out_dict = {
        "svd_input": svd_input,
        "sem_rep": P,
        "singular_values": D,
        "marker_rep": Q,
        'sem_loadings': np.multiply(P, D),
        'marker_loadings': np.multiply(D.reshape(-1, 1), Q)
    }
    
    eigenvalues, var_explained, total_var_explained = scree_plots(out_dict['singular_values'], title, data_suffix, user=user, to_show=False)
    out_dict['eigenvalues'] = eigenvalues
    out_dict['var_explained'] = var_explained
    out_dict['total_var_explained'] = total_var_explained
    out_dict['num_dim_to_keep'] = num_dim_to_keep(total_var_explained, label)
    out_dict['new_sem_rep'] = out_dict['sem_rep'][:, :out_dict['num_dim_to_keep']]
    out_dict['new_marker_rep'] = out_dict['marker_rep'].T[:, :out_dict['num_dim_to_keep']].reshape(-1, out_dict['num_dim_to_keep'])
    out_dict['new_sem_loadings'] = out_dict['sem_loadings'][:, :out_dict['num_dim_to_keep']]
    out_dict['new_marker_loadings'] = out_dict['marker_loadings'].T[:, :out_dict['num_dim_to_keep']].reshape(-1, out_dict['num_dim_to_keep'])
    return out_dict


def num_dim_to_keep(total_var_explained, label):
    """
    Computes the number of latent dimensions to keep based on the input label.
    If the input label ends with "dim", keep that number of dimensions.
    If the input label ends with "var_exp", return the number of dimensions that 
    capture the specified amount of variance explained.
    """
    label = label.split("_")
    if label[-1] == "dim":
        return int(label[-2])
    else:
        return np.argmax(total_var_explained>(int(label[-2])/100))

def plot_eigenvalues(eigenvalues, num_dim_to_show, image_dir, title, axis_offset):
    plt.plot(np.arange(num_dim_to_show) + axis_offset, eigenvalues[:num_dim_to_show])
    plt.xticks(np.arange(0, num_dim_to_show), np.arange(1, num_dim_to_show+1))

    plt.xlabel("Component Number")
    plt.ylabel("Eigenvalue (x10^6)")
    plt.title("Eigenvalues of Each Component")

    plt.savefig(f"{image_dir}/{title}_eigenvalues.jpg")
    plt.show()
    plt.clf()


def plot_variance_explained(var_explained, num_dim_to_show, image_dir, title, axis_offset):
    plt.plot(np.arange(len(var_explained[:num_dim_to_show])) + axis_offset, var_explained[:num_dim_to_show])
    plt.xticks(np.arange(0, num_dim_to_show), np.arange(1, num_dim_to_show+1))

    plt.xlabel("Component Number")
    plt.ylabel("Variance Explained")
    plt.title("Proportion of Variance Explained by Each Component")

    plt.savefig(f"{image_dir}/{title}_prop_var.jpg")
    plt.show()
    plt.clf()



def plot_cumulative_variance(total_var_explained, num_dim_to_show, image_dir, title):
    plt.plot(np.arange(-1, len(total_var_explained[:num_dim_to_show])) + 1, total_var_explained[:num_dim_to_show+1])
    plt.xticks(np.arange(0, num_dim_to_show), np.arange(1, num_dim_to_show+1))
    plt.ylim(0, np.max(total_var_explained[:num_dim_to_show+1]) + 0.02)
    
    plt.xlabel("Component Number")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Cumulative Variance Explained Across Components")

    plt.savefig(f"{image_dir}/{title}_cumulative_var.jpg")
    plt.show()
    plt.clf()



def scree_plots(sing_val_matrix, title, data_suffix, axis_offset=0, user="jai", num_dim_to_show = 10, to_show=True):
    IMAGE_DIR = f"/u/{user}/stancetaking/images/latent_dimensions/structure_exploration/{data_suffix}/{title}"
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    eigenvalues = sing_val_matrix**2

    var_explained = sing_val_matrix**2/np.sum(sing_val_matrix**2)
    total_var_explained = np.concatenate(([0], np.cumsum(var_explained)))
    var_explained = np.round(var_explained, 3)
    if to_show:
        
        print(var_explained[:num_dim_to_show])
        print(total_var_explained[:num_dim_to_show])

        plot_eigenvalues(eigenvalues[axis_offset:], num_dim_to_show, IMAGE_DIR, title, axis_offset=axis_offset)
        plot_variance_explained(var_explained[axis_offset:], num_dim_to_show, IMAGE_DIR, title, axis_offset=axis_offset)
        plot_cumulative_variance(total_var_explained, num_dim_to_show, IMAGE_DIR, title)

    return eigenvalues, var_explained, total_var_explained




def calculate_pairwise_sim(data):
    """Calculate the pairwise similarity of a vector to itself.
    
    Args:
        data: A vector
    
    Output:
        DataFrame containing pairwise similarities of each entry to each other entry
    """
    pairwise_sim = 1 - pd.DataFrame(cosine_similarity(data, data), index=data.index, columns=data.index)
    pairwise_sim = pairwise_sim.where(np.triu(np.ones(pairwise_sim.shape), k=1).astype(np.bool))
    pairwise_sim = pairwise_sim.stack().rename("Original_Dist")
    pairwise_sim.index = pairwise_sim.index.rename(["Sit 1", "Sit 2"])
    return pd.DataFrame(pairwise_sim)



# TODO: ADD TO DATA UTILS FILE
def true_total_variation_distance(need_1, need_2):
    return 0.5* np.abs(need_1 - need_2).sum()



def convert_to_cooc_matrix(data_suffix):
    print("Loading data...")
    df = Serialization.load_obj(f"stance_pipeline_full_data_{NUM_QUANTILES}_quantiles_{data_suffix}")

    print("Loading markers...")
    all_markers = sorted(df['rel_marker'].unique())
    all_markers = [marker for marker in all_markers if marker not in ["'d", "10x"]]
    df = df[df['rel_marker'].isin(all_markers)]
    # Combine the subreddit and marker and aggregate
    print("Creating sub_markers...")
    df['sub_marker'] = df["subreddit"] + "_" + df['rel_marker']
    agg = df.groupby(["bin", "sub_marker"]).count()

    print("Creating all sub_marker combinations...")
    comms = df['subreddit'].unique()
    markers = df['rel_marker'].unique()
    bins = df['bin'].unique()
    com_markers = list(product(comms, markers))
    com_markers = ["_".join(pair) for pair in com_markers]
    len(com_markers)

    # Create a matrix of all possible semantic situations and community markers with 0 values
    print("Creating full_matrix...")
    full_counts = pd.DataFrame(0, index=pd.MultiIndex.from_product([bins, com_markers], names=["bin", "sub_marker"]), columns=agg.columns)
    # Add to our attested matrix to fill in the blanks and get a full matrix
    print("Adding with existing_matrix...")
    total = agg.add(full_counts, fill_value=0)
    total = total.reset_index()
    # Create co-occurrence matrices
    print("Starting COOC Crosstab")
    cooc_matrix = pd.crosstab(total['bin'], total['sub_marker'], total['subreddit'], aggfunc="sum")
    print(f"Full Co-occurrence Matrix Size: {cooc_matrix.shape}")
    get_nonzero_prop(cooc_matrix)
    pav_matrix = pd.crosstab(df['subreddit'], df['rel_marker'])
    print(f"Pavalanathan Matrix Size: {pav_matrix.shape}")
    get_nonzero_prop(pav_matrix)

    pav_matrix = Serialization.save_obj(pav_matrix, f"pavalanathan_cooc_data_{data_suffix}") # change {full_data} to {intensifiers} to get subset
    cooc_matrix = Serialization.save_obj(cooc_matrix, f"our_cooc_data_{data_suffix}")




if __name__ == "__main__":
    #TODO: CHANGE TO CLEANER VERSION
    save_suffixes = ["may_17_intensifiers_vadpf_3_bins", "intensifiers_vaf", "intensifiers_vf"]
    markers_of_interest = load_intensifiers()
    print(markers_of_interest)
    columns_of_interests = [FEATURE_COLUMNS, ["Valence", "Arousal", "Formality"], ["Valence", "Formality"]]
    namespaces = ["VADPF", "VAF", "VF"]
    for suffix, columns_of_interest, namespace in zip(save_suffixes, columns_of_interests, namespaces):
        print(suffix)
        print(columns_of_interest)
        print(namespace)
        load_data_from_raw(suffix, markers_of_interest, namespace, columns_of_interest=columns_of_interest)
        # load_difference_data_from_raw(suffix, markers_of_interest, namespace, columns_of_interest=columns_of_interest)
        break

    convert_to_cooc_matrix("may_17_intensifiers_vadpf_3_bins")
