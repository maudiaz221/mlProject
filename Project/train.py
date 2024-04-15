import argparse
import os
from typing import Any
from PIL import Image
from common import load_image_labels, load_single_image, save_model, makeFolder
from datasets import load_dataset 
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import torch
from datasets import load_metric

########################################################################################################################

def parse_args():
    
    """
    Helper function to parse command line arguments
    :return: args object
    """
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    
    return args


def load_train_resources(resource_dir: str = 'resources') -> Any:
    
    """
    Load any resources (i.e. pre-trained models, data files, etc) here.
    Make sure to submit the resources required for your algorithms in the sub-folder 'resources'
    :param resource_dir: the relative directory from train.py where resources are kept.
    :return: TBD
    """
    path_to_models = os.path.join(resource_dir, 'models.txt')
    models = []
    with open(path_to_models, 'r') as f:
        for line in f:
            models.append(line.strip())
    
    return models 
            
            
    
    
    raise RuntimeError(
        "load_train_resources() not implement. If you have no pre-trained models you can comment this out.")
    

def pre_process(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )


    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch
    
    return preprocess_train, preprocess_val

def aquire_labels(dataset):
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):

    metric = load_metric("accuracy")
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def train(dataset, models, output_directory: str) -> Any:
    
    batch_size = 2
    label2id, id2label = aquire_labels(dataset)
    
    
    for model in models:
        image_processor  = AutoImageProcessor.from_pretrained(model)
        preprocess_train, preprocess_val = pre_process(image_processor)
        
        #split the data
        splits = dataset["train"].train_test_split(test_size=0.1)
        train_ds = splits['train']
        val_ds = splits['test']
        
        #transform the data
        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)
        
        #finetune the model
        mod = AutoModelForImageClassification.from_pretrained(
            model, 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True, 
        )
        
        model_name = model.split("/")[-1]
        direc = os.path.join(output_directory, model_name)
        
        args = TrainingArguments(
        output_dir=direc,
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5, #0.01
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        )
        
        trainer = Trainer(
        mod,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        )
        
        trainer_results = trainer.train()
        trainer.save_model()
        trainer.save_state()
        
        #finetuned_models.append(model_name)
        
        
    
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.

    return finetuned_models


def main(train_input_dir: str, train_labels_file_name: str, train_output_dir: str):
    
    
    
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """

    # load pre-trained models or resources at this stage.
    models = load_train_resources()

    # # load label file
    labels_file_path = os.path.join(train_input_dir, train_labels_file_name)
    df_labels = load_image_labels(labels_file_path)
    
    destination = makeFolder(df_labels, train_input_dir)
    
    dataset = load_dataset("imagefolder",data_dir=destination)
    
    

    # train a model for this task
    models = train(dataset, models, train_output_dir)

    


if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir
    
    os.makedirs(trained_model_output_dir, exist_ok=True)

    main(train_data_image_dir, train_data_labels_csv, trained_model_output_dir)

########################################################################################################################
