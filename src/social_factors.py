from utils.core_utils import *




def compute_distinctiveness(similarities):
    diffs = 1 - similarities
    distinctiveness =  diffs.sum()/(diffs.shape[0] - 1)
    return distinctiveness


def generate_social_factors_df():
    subreddit_agg_stats = Serialization.load_obj("2019_community_factors_num_unique_posts")
    subreddit_author_stats = Serialization.load_obj("2019_community_factors_num_unique_authors")
    loyal_users_per_subreddit = Serialization.load_obj("2019_loyal_users_per_subreddit")
    users_per_subreddit = Serialization.load_obj("2019_top_level_commentors_per_subreddit")
    density_per_subreddit = Serialization.load_obj("2019_community_factors_density") 

    contextual_distinctiveness = Serialization.load_obj("2019_contextual_distinctiveness_scores_final")
    pav_distinctiveness = Serialization.load_obj("2019_pav_distinctiveness_scores_final")

    # Convert all indices to lower case
    subreddit_agg_stats.index = subreddit_agg_stats.index.str.lower()
    subreddit_author_stats.index = subreddit_author_stats.index.str.lower()
    loyal_users_per_subreddit.index = loyal_users_per_subreddit.index.str.lower()
    users_per_subreddit.index = users_per_subreddit.index.str.lower()
    density_per_subreddit.index = density_per_subreddit.index.str.lower()

    # Loyalty variable has 7.9K communities. We need to get the subset of these 
    # communities that appear in our dataset.

    core_communities_df = contextual_distinctiveness.reset_index()
    core_communities_df.columns=['subreddit', 'dist']
    loyal_users_per_subreddit = loyal_users_per_subreddit.reset_index()
    lu_per_subreddit = core_communities_df.merge(loyal_users_per_subreddit, how='left', left_on="subreddit", right_on="subreddit")[['subreddit', 'author']].set_index("subreddit")

    # Two communities have 0 loyal users, and so do not appear in the original loyalty
    # variable. We fill these with 0.
    lu_per_subreddit = lu_per_subreddit.fillna(0)

    # Convert any DataFrames to Series
    lu_per_subreddit = lu_per_subreddit['author']
    users_per_subreddit = users_per_subreddit['author']
    density_per_subreddit = density_per_subreddit[0]

    # Reindex to all be the same
    subreddit_agg_stats = subreddit_agg_stats.loc[contextual_distinctiveness.index]
    subreddit_author_stats = subreddit_author_stats.loc[contextual_distinctiveness.index]
    lu_per_subreddit = lu_per_subreddit.loc[contextual_distinctiveness.index]
    users_per_subreddit = users_per_subreddit.loc[contextual_distinctiveness.index]
    density_per_subreddit = density_per_subreddit.loc[contextual_distinctiveness.index]

    # Compute normalized measures
    loyalty_rate = lu_per_subreddit/users_per_subreddit
    posting_rate = subreddit_agg_stats/subreddit_author_stats

    dist_df = pd.concat((contextual_distinctiveness, 
                        pav_distinctiveness, 
                        posting_rate, 
                        subreddit_author_stats, 
                        loyalty_rate, 
                        density_per_subreddit), 
                        axis=1)

    dist_df.columns = ['stance_context_distinctiveness', 
                    'marker_distinctiveness',  
                    'activity', 
                    'size', 
                    'loyalty', 
                    'density']

    dist_df = dist_df.sort_values(by='stance_context_distinctiveness')
    return dist_df

def compute_social_factor_correlations(df, representation_name):
    social_factors = ['size', 'activity', 'loyalty', 'density']
    for factor in social_factors:
        print(f"Spearman Correlation of {representation_name.capitalize()} Distinctiveness and {factor.capitalize()}")
        print(spearmanr(df[f"{representation_name}_distinctiveness"], df[factor]))
