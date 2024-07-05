import numpy as np
import torch
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from fire import Fire
from transformers import CLIPFeatureExtractor, CLIPImageProcessor

from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel

from torch import nn, einsum

from trainer.models.base_model import BaseModelConfig

from transformers import CLIPConfig
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from typing import Any, Optional, Tuple, Union
import torch
import cv2
import os

from trainer.models.cross_modeling import Cross_model
import matplotlib.pyplot as plt
import torch.nn.functional as F

import gc
import json


@torch.no_grad()

def infer_one_sample(image, prompt, clip_model, clip_processor, tokenizer, device, condition=None):
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    image_input = _process_image(image).to(device)
    text_input = _tokenize(prompt).to(device)
    if condition is None:
        condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
    condition_batch = _tokenize(condition).repeat(text_input.shape[0],1).to(device)

    with torch.no_grad():
        text_f, text_features = clip_model.model.get_text_features(text_input)

        image_f = clip_model.model.get_image_features(image_input.half())
        condition_f, _ = clip_model.model.get_text_features(condition_batch)

        sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
        mask = mask.repeat(1,image_f.shape[1],1)
        image_features = clip_model.cross_model(image_f, text_f,mask.half())[:,0,:]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_score = clip_model.logit_scale.exp() * text_features @ image_features.T

    return image_score[0]

def infer_example(images, prompt, clip_model, clip_processor, tokenizer, device):
    scores = []
    for image in images:
        score = infer_one_sample(image, prompt, clip_model, clip_processor, tokenizer, device)
        scores.append(score)
    scores = torch.stack(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)[0]
    return probs.cpu().tolist()

def acc(score_sample, predict_sample):
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(score_sample)):
        item_base = score_sample[idx]["rank"]
        item = predict_sample[idx]["rewards"]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    return true_cnt / tol_cnt

def inversion_score(predict_sample, score_sample):
    n = len(score_sample)
    cnt = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if score_sample[i] > score_sample[j] and predict_sample[i] > predict_sample[j]:
                cnt += 1
            elif score_sample[i] < score_sample[j] and predict_sample[i] < predict_sample[j]:
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)

def main():
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    device = "cuda"
    image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)

    model_ckpt_path = "outputs/MPS_overall_checkpoint.pth"
    model = torch.load(model_ckpt_path)
    model.eval().to(device)

    score_sample = []
    with open("hpdv2/test.json", "r") as f:
        score_sample = json.load(f)
    
    predict_sample = []
    score = 0.
    with torch.no_grad():
        for i in range(len(score_sample)):
            item = score_sample[i]
            rewards = infer_example(item["image_path"], item["prompt"], model, image_processor, tokenizer, device)
            score += inversion_score(rewards, item['rank'])
    test_acc = score / len(score_sample)
    print(f"HPDv2 Test Acc: {100 * test_acc:.2f}%")


if __name__ == '__main__':
    Fire(main)
