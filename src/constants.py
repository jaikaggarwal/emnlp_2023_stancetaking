DATA_SUFFIX = "may_17_intensifiers_vadpf_3_bins"
FEATURE_COLUMNS = ['Valence', 'Arousal', 'Dominance', 'Politeness', 'Formality']
LINGUISTIC_PROPERTIES = FEATURE_COLUMNS + [col + "_Absolute" for col in FEATURE_COLUMNS]
