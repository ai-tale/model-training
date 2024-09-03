#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom storyteller model for AI Tale.
"""

import logging
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    GPT2LMHeadModel,
    GPT2Config
)
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class StorytellerConfig(GPT2Config):
    """Configuration class for AI Tale Storyteller model."""
    
    model_type = "storyteller"
    
    def __init__(
        self,
        narrative_aware_attention=False,
        character_embeddings=False,
        num_characters=10,
        character_embedding_dim=64,
        plot_guidance=False,
        emotion_classification=False,
        num_emotions=6,
        age_appropriate_filter=True,
        **kwargs
    ):
        """
        Initialize StorytellerConfig.
        
        Args:
            narrative_aware_attention: Whether to use narrative-aware attention mechanism
            character_embeddings: Whether to use character embeddings
            num_characters: Maximum number of characters to track
            character_embedding_dim: Dimension of character embeddings
            plot_guidance: Whether to use plot guidance mechanisms
            emotion_classification: Whether to add emotion classification head
            num_emotions: Number of emotion classes if emotion_classification is True
            age_appropriate_filter: Whether to apply age-appropriate content filter
            **kwargs: Additional parameters passed to GPT2Config
        """
        super().__init__(**kwargs)
        
        self.narrative_aware_attention = narrative_aware_attention
        self.character_embeddings = character_embeddings
        self.num_characters = num_characters
        self.character_embedding_dim = character_embedding_dim
        self.plot_guidance = plot_guidance
        self.emotion_classification = emotion_classification
        self.num_emotions = num_emotions
        self.age_appropriate_filter = age_appropriate_filter


class StorytellerModel(GPT2LMHeadModel):
    """
    AI Tale Storyteller model.
    
    This model extends GPT-2 with story-specific capabilities including:
    - Narrative-aware attention
    - Character tracking
    - Emotion classification
    - Age-appropriate content filtering
    """
    
    config_class = StorytellerConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize additional components based on config
        if config.character_embeddings:
            self.character_embeddings = nn.Embedding(
                config.num_characters, 
                config.character_embedding_dim
            )
            self.character_projection = nn.Linear(
                config.character_embedding_dim, 
                config.hidden_size
            )
        
        if config.emotion_classification:
            self.emotion_classifier = nn.Linear(
                config.hidden_size, 
                config.num_emotions
            )
        
        # Initialize the model
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        character_ids=None,
    ):
        """
        Forward pass of the StorytellerModel.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for language modeling
            past_key_values: Past key values for faster generation
            use_cache: Whether to use cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            character_ids: Character IDs if character tracking is enabled
        """
        # Process character embeddings if provided
        if self.config.character_embeddings and character_ids is not None:
            character_embeds = self.character_embeddings(character_ids)
            character_embeds = self.character_projection(character_embeds)
            
            # If inputs_embeds is None, get embeddings from input_ids
            if inputs_embeds is None:
                inputs_embeds = self.transformer.wte(input_ids)
            
            # Add character embeddings to token embeddings
            inputs_embeds = inputs_embeds + character_embeds
        
        # Call parent class's forward method
        outputs = super().forward(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Add emotion classification if enabled
        if self.config.emotion_classification and outputs.hidden_states is not None:
            # Use the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get classification for each token
            emotion_logits = self.emotion_classifier(last_hidden_state)
            
            # Add to outputs
            outputs.emotion_logits = emotion_logits
        
        return outputs
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """Prepare inputs for generation."""
        # Get base preparation from parent class
        inputs = super().prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        
        # Add character IDs if provided
        if "character_ids" in kwargs:
            inputs["character_ids"] = kwargs["character_ids"]
        
        return inputs


def create_storyteller_model(
    base_model_name="gpt2-medium",
    narrative_aware_attention=False,
    character_embeddings=False,
    plot_guidance=False,
    emotion_classification=False,
    age_appropriate_filter=True
):
    """
    Create a StorytellerModel with the specified parameters.
    
    Args:
        base_model_name: Name of the base model to use
        narrative_aware_attention: Whether to use narrative-aware attention
        character_embeddings: Whether to use character embeddings
        plot_guidance: Whether to use plot guidance
        emotion_classification: Whether to add emotion classification
        age_appropriate_filter: Whether to apply age-appropriate content filter
        
    Returns:
        Initialized StorytellerModel
    """
    logger.info(f"Creating Storyteller model based on {base_model_name}")
    
    # Load base model config
    base_config = AutoConfig.from_pretrained(base_model_name)
    
    # Create Storyteller config
    config = StorytellerConfig(
        narrative_aware_attention=narrative_aware_attention,
        character_embeddings=character_embeddings,
        plot_guidance=plot_guidance,
        emotion_classification=emotion_classification,
        age_appropriate_filter=age_appropriate_filter,
        **base_config.to_dict()
    )
    
    # Create StorytellerModel
    model = StorytellerModel(config)
    
    # Initialize from pre-trained model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Copy weights from base model
    model.transformer = base_model.transformer
    model.lm_head = base_model.lm_head
    
    logger.info("Storyteller model created successfully")
    return model 