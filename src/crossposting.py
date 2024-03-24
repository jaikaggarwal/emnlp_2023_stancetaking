from utils.core_utils import *


def generate_crossposting_predictions(name_to_similarities):
    """
    Generate each representation's predictions for how much crossposting
    occurs between a pair of communities. 
    Args:
        -- name_to_similarities (dict): keys are representation names, values are 
        dataframes of pairwise similarities
    Output:
        -- dict: keys are representation names, values are list of similarity values
    """
    crossposting_df = pd.read_csv("../data_files/crossposting_full_community_data_2019.csv")
    crossposting_df['com_1'] = crossposting_df['com_1'].str.lower()
    crossposting_df['com_2'] = crossposting_df['com_2'].str.lower()

    for name in name_to_similarities:
        curr_similarities = name_to_similarities[name]
        crossposting_similarities = []

        for _, row in crossposting_df.iterrows():
            com_1, com_2 = row['com_1'], row['com_2']
            crossposting_similarities.append(curr_similarities.loc[com_1][com_2])
        
        crossposting_df[name] = crossposting_similarities
    
    return crossposting_df