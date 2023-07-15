from torch.utils.data import Dataset


class Event_trigger_dataset(Dataset):
    def __init__(
            self,
            all_tokens,
            all_segment_ids,
            all_mask_ids,
            all_labels,
            example_ids,
            all_trigger_masks,
    ):
        self.all_tokens = all_tokens
        self.all_segment_ids = all_segment_ids
        self.all_mask_ids = all_mask_ids
        self.all_labels = all_labels
        self.example_ids = example_ids
        self.all_trigger_masks = all_trigger_masks

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],
            self.all_segment_ids[idx],
            self.all_mask_ids[idx],
            self.all_labels[idx],
            self.example_ids[idx],
            self.all_trigger_masks[idx],
        )

    def __len__(self):
        return len(self.all_tokens)


class QG_dataset(Dataset):
    def __init__(self, all_input_ids, all_mask_ids,
                 all_target_ids, all_feature_idex):
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_target_ids = all_target_ids
        self.all_feature_idex = all_feature_idex

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_target_ids[idx], self.all_feature_idex[idx]

    def __len__(self):
        return len(self.all_input_ids)


class QG_dataset_inference(Dataset):
    def __init__(self, all_input_ids, all_mask_ids,
                 all_feature_idex):
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_feature_idex = all_feature_idex

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_feature_idex[idx]

    def __len__(self):
        return len(self.all_input_ids)


class QA_dataset(Dataset):
    def __init__(self, all_input_ids, all_mask_ids,
                 all_target_ids, all_feature_idex):
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_target_ids = all_target_ids
        self.all_feature_idex = all_feature_idex

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_target_ids[idx], self.all_feature_idex[idx]

    def __len__(self):
        return len(self.all_input_ids)
