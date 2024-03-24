from crossposting import *
from social_factors import *
from representations import *



def main():

    with open("../data_files/list_of_communities.txt", "r") as file:
        communities = json.load(file)

    marker_rep = load_marker_representations(communities, num_dim_to_keep=11)
    stance_context_rep = load_stance_context_representations(communities)
    textual_rep = load_textual_representations(communities, num_dim_to_keep=7)

    marker_similarity = compute_pairwise_sims(marker_rep, communities)
    stance_context_similarity = compute_pairwise_sims(stance_context_rep, communities)
    textual_similarity = compute_pairwise_sims(textual_rep, communities)

    # Compute correlations of representations
    marker_unique_values = get_unique_pairwise_sims(marker_similarity)
    stance_context_unique_values = get_unique_pairwise_sims(stance_context_similarity) 
    textual_unique_values = get_unique_pairwise_sims(textual_similarity) 

    print("Correlation of Stance Marker and Stance Context Representations")
    print(pearsonr(marker_unique_values, stance_context_unique_values))

    print("Correlation of Stance Marker and Textual Representations")
    print(pearsonr(marker_unique_values, textual_unique_values))

    print("Correlation of Stance Context and Textual Representations")
    print(pearsonr(stance_context_unique_values, textual_unique_values))
    

    # TODO: Add the graphs in Section 5
    sub_to_topic, sub_to_special_topic, topic_to_special_topic = load_topics()

    # Crossposting analysis
    name_to_similarities = {"marker": marker_similarity, 
                            "stance_context": stance_context_similarity}
    predictions = generate_crossposting_predictions(name_to_similarities)
    
    # The analyses are performed in R, so we save the outputs
    predictions.to_csv("../data_files/crossposting_regression_emnlp_validation_2024.csv", index=False)


    # Social factors analysis
    social_factors_df = generate_social_factors_df()
    compute_social_factor_correlations(social_factors_df, 'marker')
    compute_social_factor_correlations(social_factors_df, 'stance_context')

    

if __name__ == "__main__":
    main()