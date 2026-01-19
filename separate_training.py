# 1. We read first datasets/cleaned_lightpath_dataset.csv and datasets/cleaned_lightpath_target.csv
# 2. We separate the dataset into training, testing and validation sets. (80/10/10)
# 3. We print that we reached this step.
# 4. we cdefine a TabularDataset(Dataset) pytorch class with len and get item for our data
# 5. we create a function named get_dataloader that takes X, y, batch_size, and sampler_type={random, m_per_class, class_balanced}, and also m_per_class= number if this is selected, and shuffle boolean.
# This function will be used to first select the sampling strategy then return DataLoader instance.
# The MPerClassSampler exists in pytorch, for class_balanced here is how to deal with it:
#    elif sampler_type == "class_balanced":
# # Weighted sampling based on inverse class frequency
# class_counts = np.bincount(y_np)
# class_weights = 1. / class_counts
# weights = torch.DoubleTensor([class_weights[label] for label in y_np])
# sampler = WeightedRandomSampler(
#     weights=weights,
#     num_samples=len(weights),
#     replacement=True,
#     generator=torch.Generator().manual_seed(seed)  # Important for reproducibility
# )
# shuffle = False

# Shuffle is deactivated when using samplers. activated in random sampling
# So in other words, sampling strategy and batch size and m_per_class are hyperparameters to investigate
# 6. we defined the train loaders, val loaders, and test loaders
# 7.
