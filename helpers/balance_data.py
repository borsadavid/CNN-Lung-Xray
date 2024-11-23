import pandas as pd
from sklearn.utils import resample

def preprocess_and_balance_data(data_entry_file, labels_map, max_entries):
    """
    Preprocess and strictly balance the dataset by excluding multi-label rows
    and balancing all labels to the average count.

    Args:
        data_entry_file (str): Path to the CSV file with data entries.
        labels_map (dict): Mapping of valid labels.
        max_entries (int): Maximum number of rows to process.

    Returns:
        pd.DataFrame: Strictly balanced dataset.
    """
    # Load the first `max_entries` rows
    data = pd.read_csv(data_entry_file).head(max_entries)

    # Expand each row for multi-label cases and filter rows with valid labels
    data['Finding_Labels_List'] = data['Finding Labels'].str.split('|').apply(lambda x: [l.strip() for l in x])
    data = data[data['Finding_Labels_List'].apply(lambda labels: any(label in labels_map for label in labels))]

    # Exclude rows with more than one label
    data = data[data['Finding_Labels_List'].apply(len) == 1]

    # Simplify `Finding_Labels_List` to a single label (convert to string)
    data['Finding_Label'] = data['Finding_Labels_List'].apply(lambda x: x[0])

    # Calculate label counts
    label_counts = data['Finding_Label'].value_counts().to_dict()

    # Compute the target count (average count of all labels)
    avg_count = int(sum(label_counts.values()) / len(label_counts))

    # Create a balanced dataset
    balanced_data = []
    for label, count in label_counts.items():
        label_data = data[data['Finding_Label'] == label]
        if len(label_data) < avg_count:
            # Upsample
            label_data = resample(
                label_data,
                replace=True,
                n_samples=avg_count,
                random_state=42
            )
        elif len(label_data) > avg_count:
            # Downsample
            label_data = resample(
                label_data,
                replace=False,
                n_samples=avg_count,
                random_state=42
            )
        balanced_data.append(label_data)

    # Combine balanced data and shuffle
    balanced_data = pd.concat(balanced_data, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data
