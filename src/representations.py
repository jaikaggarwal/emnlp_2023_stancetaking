from utils.data_utils import *
from utils.topic_utils import *
from constants import DATA_SUFFIX, FEATURE_COLUMNS, LINGUISTIC_PROPERTIES


def load_marker_representations(communities, num_dim_to_keep):
    # Load Pavalanathan data and associated global variables
    pav_matrix = Serialization.load_obj(f"pavalanathan_cooc_data_{DATA_SUFFIX}")
    pav_matrix = pav_matrix.loc[communities]
    intensifiers = pav_matrix.columns.tolist()

    print(f"Num markers: {len(intensifiers)}")

    # Perform SVD on Pavalanathan data
    pav_svd_output = generate_latent_representations(
        matrix=pav_matrix, 
        label="pav_ppmi_95_var", 
        title="Pavalanathan",
        data_suffix=DATA_SUFFIX,
        include_ppmi=True
    )

    # Visualize scree plot to find elbow
    scree_plots(pav_svd_output['singular_values'][:], "Marker Representation", DATA_SUFFIX, 
                axis_offset=1, num_dim_to_show=15)
    pav_latent_representation = pav_svd_output['new_sem_loadings'][:,:num_dim_to_keep]
    return pd.DataFrame(pav_latent_representation, index=communities, columns=np.arange(num_dim_to_keep))



def load_stance_context_representations(communities):
    """
    We pass in a list of communities to ensure that the order of the rows of each representation
    are aligned.
    """
    # Load data frames
    print("Load dataframe..")
    sentence_df = Serialization.load_obj(f"stance_pipeline_full_data_with_sentences_{DATA_SUFFIX}")
    print(sentence_df.shape)

    # Compute extremeness
    print("Compute extremeness...")
    sentence_df[[col + "_Absolute" for col in FEATURE_COLUMNS]] = np.abs(sentence_df[FEATURE_COLUMNS] - sentence_df[FEATURE_COLUMNS].mean())

    # Creating community-level contextual representations
    print("Creating community representations...")
    community_representation = sentence_df[['subreddit'] + LINGUISTIC_PROPERTIES].groupby(['subreddit']).mean()
    community_representation = community_representation.loc[communities]
    centered_community_representation = (community_representation - community_representation.mean(axis=0))
    rescaled_community_representation = centered_community_representation / community_representation.std(axis=0)

    return rescaled_community_representation


def load_textual_representations(communities, num_dim_to_keep):
    textual_svd = Serialization.load_obj("textual_tf_idf_svd_sentence_df")
    
    # Get indices of dimensions that explain the most variance
    most_important_dimensions = np.argsort(textual_svd.explained_variance_ratio_)[::-1][:num_dim_to_keep]
    
    textual_representations = np.load('/ais/hal9000/datasets/reddit/brian_unigrams/tf_idf_storage/svd-tf-idf_unigrams_sentence_df.npy')[:, most_important_dimensions]
    textual_representation_rows = pd.read_csv('/ais/hal9000/datasets/reddit/brian_unigrams/tf_idf_storage/unigram_rows_sentence_df', header=None)
    textual_representation_df = pd.DataFrame(textual_representations, index=textual_representation_rows[0])

    # communities_with_data = (Serialization.load_obj("luo_data_2019_10k_sample")['subreddit'].unique(), textual_representation_df.index)
    textual_representation_df = textual_representation_df.loc[communities]
    return textual_representation_df


def compute_pairwise_sims(rep, communities):
    similarity_values = cosine_similarity(rep)
    return pd.DataFrame(similarity_values, index=communities, columns=communities)
    

def get_unique_pairwise_sims(pairwise_df):
    unique_values = pairwise_df.to_numpy()[np.triu_indices(pairwise_df.shape[0], k=1)]
    return unique_values

