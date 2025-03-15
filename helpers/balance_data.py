import pandas as pd
from sklearn.utils import resample


def preprocess_data(data_entry_file, labels_map, max_entries=None, random_sample=True):
    data = pd.read_csv(data_entry_file)

    # If max_entries is specified and less than the total number of rows, select a subset.
    if max_entries is not None and max_entries < len(data):
        if random_sample:
            data = data.sample(n=max_entries, random_state=42)
        else:
            data = data.head(max_entries)

    # Split the 'Finding Labels' string into a list of labels.
    data['Finding_Labels_List'] = (
        data['Finding Labels']
        .str.split('|')
        .apply(lambda x: [label.strip() for label in x])
    )

    # Filter rows to keep only those with at least one label in labels_map.
    data = data[data['Finding_Labels_List'].apply(lambda labels: any(lbl in labels_map for lbl in labels))]

    # Create a multi-hot encoding for each label in labels_map.
    for label in labels_map.keys():
        data[label] = data['Finding_Labels_List'].apply(lambda lbls: 1 if label in lbls else 0)

    return data

def balance_and_upsample_data(train_data, labels_map):
    # Compute the maximum count among all labels.
    max_count = max([train_data[label].sum() for label in labels_map.keys()])

    # For each label, if its count is less than max_count, upsample its rows.
    for label in labels_map.keys():
        current_count = train_data[label].sum()
        if current_count < max_count:
            # Compute factor as the floor of (max_count / current_count) capped at 3.
            factor = int(min(3, max_count // current_count))
            if factor > 1:
                # Calculate the number of additional samples required.
                num_to_add = current_count * (factor - 1)
                # Select rows where this label is present.
                label_rows = train_data[train_data[label] == 1]
                # Upsample these rows with replacement.
                upsampled = resample(label_rows, replace=True, n_samples=num_to_add, random_state=42)
                # Append the upsampled rows.
                train_data = pd.concat([train_data, upsampled], axis=0)

    # Shuffle the data after upsampling.
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_data
