#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics utility functions for evaluating story generation models.
"""

import logging
import re
import string
from typing import Dict, List, Tuple, Union

import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from sacrebleu import BLEU

# Set up logging
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics for generated stories.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
        
    Returns:
        Dictionary with various metrics
    """
    metrics = {}
    
    # BLEU score
    metrics.update(compute_bleu(references, hypotheses))
    
    # ROUGE scores
    metrics.update(compute_rouge(references, hypotheses))
    
    # Lexical diversity
    metrics.update(compute_lexical_diversity(hypotheses))
    
    # Story specific metrics
    metrics.update(compute_narrative_metrics(hypotheses))
    
    return metrics


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute BLEU score."""
    bleu = BLEU()
    
    # SacreBLEU expects a list of references for each hypothesis
    references_list = [[ref] for ref in references]
    
    try:
        score = bleu.corpus_score(hypotheses, references_list)
        return {"bleu": score.score / 100.0}  # Convert to 0-1 range
    except Exception as e:
        logger.warning(f"Error computing BLEU score: {e}")
        return {"bleu": 0.0}


def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        try:
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except Exception as e:
            logger.warning(f"Error computing ROUGE scores for a sample: {e}")
    
    if not rouge1_scores:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
    
    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }


def compute_lexical_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Compute lexical diversity metrics for generated texts.
    - Type-Token Ratio (TTR): ratio of unique words to total words
    - Mean Word Length: average length of words
    - Sentence Length Variation: standard deviation of sentence lengths
    """
    # Initialize metrics
    total_ttr = []
    total_mean_word_length = []
    total_sent_length_variation = []
    
    for text in texts:
        # Tokenize
        tokens = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Clean tokens (remove punctuation)
        clean_tokens = [token for token in tokens if token not in string.punctuation]
        
        # Type-Token Ratio
        if clean_tokens:
            unique_tokens = set(clean_tokens)
            ttr = len(unique_tokens) / len(clean_tokens)
            total_ttr.append(ttr)
        
        # Mean Word Length
        if clean_tokens:
            mean_word_length = np.mean([len(token) for token in clean_tokens])
            total_mean_word_length.append(mean_word_length)
        
        # Sentence Length Variation
        if sentences:
            sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
            if len(sent_lengths) > 1:
                variation = np.std(sent_lengths)
                total_sent_length_variation.append(variation)
    
    results = {}
    
    if total_ttr:
        results["type_token_ratio"] = np.mean(total_ttr)
    
    if total_mean_word_length:
        results["mean_word_length"] = np.mean(total_mean_word_length)
    
    if total_sent_length_variation:
        results["sentence_length_variation"] = np.mean(total_sent_length_variation)
    
    return results


def compute_narrative_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute metrics specific to narrative texts:
    - Dialogue Ratio: proportion of the text that is dialogue
    - Emotional Language: count of emotional terms
    - Character Presence: count of potential character indicators
    """
    # Load emotional words (simplified example)
    emotion_words = set([
        "happy", "sad", "angry", "scared", "surprised", "disgusted", 
        "joy", "sorrow", "rage", "fear", "shock", "disgust",
        "love", "hate", "excited", "worried", "terrified", "delighted",
        "crying", "laughing", "screaming", "smiling", "frowning"
    ])
    
    total_dialogue_ratio = []
    total_emotion_density = []
    total_character_density = []
    
    for text in texts:
        # Dialogue ratio (simplified - looks for quotation marks)
        dialogue_chars = len(re.findall(r'["\'](.*?)["\']', text))
        total_chars = len(text)
        if total_chars > 0:
            dialogue_ratio = dialogue_chars / total_chars
            total_dialogue_ratio.append(dialogue_ratio)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        clean_tokens = [token for token in tokens if token not in string.punctuation]
        
        if clean_tokens:
            # Emotion word density
            emotion_count = sum(1 for token in clean_tokens if token in emotion_words)
            emotion_density = emotion_count / len(clean_tokens)
            total_emotion_density.append(emotion_density)
            
            # Character presence (simplified - looks for capitalized words not at the beginning of sentences)
            sentences = sent_tokenize(text)
            character_indicators = 0
            
            for sentence in sentences:
                words = sentence.strip().split()
                if len(words) > 1:  # Skip the first word which is naturally capitalized
                    for word in words[1:]:
                        if word and word[0].isupper():
                            character_indicators += 1
            
            character_density = character_indicators / len(clean_tokens)
            total_character_density.append(character_density)
    
    results = {}
    
    if total_dialogue_ratio:
        results["dialogue_ratio"] = np.mean(total_dialogue_ratio)
    
    if total_emotion_density:
        results["emotion_density"] = np.mean(total_emotion_density)
    
    if total_character_density:
        results["character_density"] = np.mean(total_character_density)
    
    return results


def compute_creative_metrics(texts: List[str]) -> Dict[str, float]:
    """Calculate metrics that might indicate creativity or originality."""
    # To be implemented with more sophisticated creativity measures
    # This could include rare word usage, narrative complexity, etc.
    return {"creative_score": 0.5}  # Placeholder 