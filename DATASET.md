# Dataset

Please refer to [DATASET.md](DATASET.md) for downloading datasets and constructing your own dataset.

## ICRT-MT Dataset
We host the ICRT-MT, a multi-task dataset that we used to train ICRT, on [🤗HuggingFace](https://huggingface.co/datasets/Ravenh97/ICRT-MT). Download them and update [dataset_config.json](config/dataset_config.json) so that the files points to the correct locations.
```bash 
# install git-lfs
sudo apt install git-lfs
git lfs install
# clone the dataset
git clone git@hf.co:datasets/Ravenh97/ICRT-MT
# or you can download the files manually from here: https://huggingface.co/datasets/Ravenh97/ICRT-MT
cd ICRT-MT
```

## Converted DROID Dataset
Coming soon!

## Use Your Own Dataset (Collected on a DROID Setup)
We collected the dataset on the [DROID](https://github.com/droid-dataset/droid) platform. To run the conversion, please install all the dependencies (i.e. ZED) and DROID. 

After recording the demonstrations for each task (i.e. 50 demonstrations of `task_a` in `folder_a`, and 50 demonstrations of `task_b` in `folder_b`), create a new folder `training_dataset` and move all the task folders into this new folder. 

We convert the dataset via the following two scripts:
```bash
# Saving images from ZED SVO files to hdf5
python tools/preprocess_droid.py --folder /path/to/data/training_dataset

# Merging task specific hdf5 files and generate meta data to data_output
python tools/convert_droid_icrt.py --droid_dir /path/to/data/training_dataset --out_dir /path/to/save_dir
```

The converted dataset is stored in the folder `/path/to/save_dir`, which has the following structure:
```bash
/path/to/save_dir/
│
├── epi_len_mapping.json # {episode_name : length of the demonstration}
├── hdf5_keys.json # [a list of episode_names]
├── r2d2.hdf5 # an hdf5 that contains image observations, robot's state and action information
└── verb_to_episode.json # based on how trajectories are grouped in /path/to/data/training_dataset, this is a mapping from {task_name : [a list of trajectories]}
```

The key structure of r2d2.hdf5 can be parsed as the following:

```
r2d2.hdf5 
│ 
└── trajectory_1
│   ├── observation
│   │   ├── wrist_image_left 
│   │   │   └── (A list of buffers. Upon decoding, can be reshaped to (180, 320, 3))
│   │   ├── exterior_image_1_left 
│   │   │   └── (A list of buffers. Upon decoding, can be reshaped to (180, 320, 3))
│   │   ├── cartesian_position 
│   │   │   └── (A list of cartesian positions and Euler XYZ rotations (x, y, z, r, p, y))
│   │   └── gripper_position 
│   │       └── (A list of continuous gripper positions, with 0 as open and 1 as closed)
│   │
│   └── action
│       ├── cartesian_position 
│       │   └── (A list of cartesian positions and Euler XYZ rotations (x, y, z, r, p, y))
│       └── gripper_position 
│           └── (A list of continuous gripper positions, with 0 as open and 1 as closed)
│
└── trajectory_2
    ... 
```

Then update/create a copy of the [dataset config](config/dataset_config_template.json) by providing the path to each of the listed file. 

## Advanced Usage
### Merge Multiple Datasets
If you want to merge multiple datasets, there are a few changes that need to be made to the [dataset config](config/dataset_config_template.json): 
1. Update "dataset_path" with a list path to the different `r2d2.hdf5`
2. Update "hdf5_keys" in the same order as "dataset_path"
3. Merge all `epi_len_mapping_icrl.json` from each dataset into a single json file (i.e. `epi_len_mapping_icrl_merged.json`). The json file should be a single dictionary of the form `{episode_name : length of the demonstration}`. 
4. Merge all `verb_to_episode.json` from each dataset into a single json file (i.e. `verb_to_episode_merged.json`). The json file should contain a single dictionary of the form `{task_name : [a list of trajectories]}`.
Please refer to [this dataset config](config/dataset_config.json) for an example. 

### Balancing Action Primitives
While the ICRT model does not explicitly model action primitives, it is still helpful for us as researchers to understand the composition of the dataset. As presented in the paper, each task is defined by the action primitive (i.e. pick-and-place, open drawer) and what objects the action primitive is acted on. An example of a task is `pick up the toy tiger and place in the bowl`, or `poke the toy radish`. Just like to train [large language models](https://arxiv.org/abs/2307.09288), you may want to have more control over the distribution of the training tasks. We provide a way to do so by defining `task_grouping`: an association between primitives and their corresponding tasks. We can further control how often each action primitive is sampled during each epoch of training. An example is shown in the [task_grouping.json](config/data_config/icrt_mt/task_grouping.json), which is used by [this dataset config](config/dataset_config.json).
