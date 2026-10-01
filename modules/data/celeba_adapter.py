"""Bridge image datasets into the shared trainer's (x, y, group) contract."""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from modules.utils.dataset_utils import DatasetSpec, LoadedDataset


class SupervisedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        row = self.dataset[index]
        return row['image'], row['label'], row['group_id']


def prepare_image_data(config):
    from modules.data.CelebA import CelebAConfig, build_celeba_datasets
    from modules.pipelines.train_predictive_pipeline import PreparedData
    if config.append_protected_to_predictor or config.build_px_loaders:
        raise ValueError('CelebA images require append_protected_to_predictor=false and build_px_loaders=false')
    datasets = build_celeba_datasets(CelebAConfig(**dict(config.dataset_kwargs)))
    frames, offset, splits = [], 0, {}
    for name, ds in zip(('train','validation','test'), datasets):
        frame = pd.DataFrame({'y':ds.labels.numpy(), 'group_id':ds.group_ids.numpy()})
        frame['group'] = [ds.group_names[g] for g in frame.group_id]
        for attr in ds.group_attrs:
            from modules.data.CelebA import CELEBA_ATTR_TO_IDX
            frame[attr] = ds.attrs_01[:,CELEBA_ATTR_TO_IDX[attr]].numpy()
        frame.index = np.arange(offset, offset+len(frame))
        splits[name] = frame.index.to_numpy()
        frames.append(frame)
        offset += len(frame)
    spec = DatasetSpec(name='celeba', protected_cols=datasets[0].group_attrs, label_col='y')
    loaded = LoadedDataset(df=pd.concat(frames), original_columns=list(frames[0].columns),
        group_id={name:i for i,name in enumerate(datasets[0].group_names)},
        metadata={'spec':spec,'split_indices':splits})
    loaders = [DataLoader(SupervisedImages(ds), batch_size=config.loader.batch_size if i==0 else config.loader.eval_batch_size,
                         shuffle=config.loader.shuffle_train if i==0 else False)
               for i,ds in enumerate(datasets)]
    return PreparedData(loaded=loaded,spec=spec,train_df=frames[0],val_df=frames[1],test_df=frames[2],
        predictor_schema={'modality':'image','spec':spec,'dataset_kwargs':dict(config.dataset_kwargs)},
        train_loader=loaders[0],val_loader=loaders[1],test_loader=loaders[2],
        num_features=3,num_classes=2,binary=config.train.binary)
