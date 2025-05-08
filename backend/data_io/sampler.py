import torch

def participant_kfold_split(dataset, n_splits=5, shuffle=True, random_state=None):
    """
    Returns K folds with participant-wise separation
    
    Args:
        dataset: Your dataset object (must have labels/mat_files accessible)
        n_splits: Number of folds (default: 5)
        shuffle: Whether to shuffle participants (default: True)
        random_state: Seed for reproducibility (default: None)
        
    Returns:
        List of K folds, each containing (train_subset, test_subset)
    """
    torch.manual_seed(4)
    # Get unique participants
    participants = list(dataset.labels.keys())
    num_participants = len(participants)
    print(str(dataset.mat_files[0])[18])
    # Shuffle participants if needed
    if n_splits == 1:
        train_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in participants
        ]
        return [(
            torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, []),  # Empty test set
            participants,         # All participants in train
            []                   # No test participants
        )]


    if shuffle:
        if random_state is not None:
            torch.manual_seed(random_state)
        indices = torch.randperm(num_participants).tolist()
        participants = [participants[i] for i in indices]
    # Create folds
    folds = []
    for fold in range(n_splits):
        # Calculate test participant range for this fold
        fold_size = num_participants // n_splits
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold != n_splits - 1 else num_participants
        
        # Split participants
        test_participants = participants[test_start:test_end]
        train_participants = [p for p in participants if p not in test_participants]
        
        # Get sample indices
        train_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in train_participants
        ]
        test_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in test_participants
        ]
        
        folds.append((
            torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, test_indices),
            train_participants,  # Optional: return participant lists
            test_participants    # for tracking
        ))

    return folds