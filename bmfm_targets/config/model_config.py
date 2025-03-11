from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from bmfm_targets.config import FieldInfo, LabelColumnInfo

logger = logging.get_logger(__name__)

PerformerKernel = Enum("PerformerKernel", ["cosh", "exp", "elu", "relu"])
FeatureGenerationAlgorithm = Enum(
    "FeatureGenerationAlgorithm", ["auto", "kacs", "qr", "guassian"]
)


class SCModelConfigBase(PretrainedConfig):
    def to_dict(self):
        """Serializes class to a Python dictionary."""
        output = super().to_dict()
        if self.fields is not None:
            output["fields"] = [fi.to_dict() for fi in self.fields]
        else:
            output["fields"] = None

        if self.label_columns is not None:
            output["label_columns"] = [fi.to_dict() for fi in self.label_columns]
        else:
            output["label_columns"] = None
        output = {k: v.name if isinstance(v, Enum) else v for k, v in output.items()}
        return output

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "PretrainedConfig":
        fields = [FieldInfo(**fi) for fi in config_dict["fields"]]
        label_columns = (
            [LabelColumnInfo(**fi) for fi in config_dict["label_columns"]]
            if config_dict.get("label_columns")
            else None
        )
        config_dict["fields"] = fields
        config_dict["label_columns"] = label_columns
        return super().from_dict(config_dict, **kwargs)

    def __setstate__(self, state):
        state.setdefault("label_columns", None)
        self.__dict__.update(state)


class SCBertConfig(SCModelConfigBase):
    """
    Configuration class to store the configuration of a [`SCBertModel`].

    It is used to instantiate a BERT model according to the
    specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that
    of the BERT architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.


    Attributes
    ----------
        fields (`list[FieldInfo]`):
            A list of `FieldInfo` objects, each representing a field.
        checkpoint (torch.serialiaziation.FILE_LIKE): checkpoint to load from
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    """

    model_type = "scbert"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: torch.serialization.FILE_LIKE | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
        classifier_dropout: float | None = None,
        max_position_embeddings: int = 512,
        position_embedding_type: str | None = None,
        attention: str | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Args:
        ----
        fields (`list[FieldInfo]`):
        A list of `FieldInfo` objects, each representing a field.
        num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

        """
        self.attention = attention
        self.attention_kwargs = attention_kwargs

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.fields = fields
        self.label_columns = label_columns
        self.checkpoint = checkpoint
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type


class SCPerformerConfig(SCModelConfigBase):
    """
    Configuration class to store the configuration of a [`SCPerformerModel`].

    It is used to instantiate a BERT model according to the
    specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that
    of the BERT architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.

    """

    model_type = "scperformer"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: torch.serialization.FILE_LIKE | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
        kernel_type: str | Callable | PerformerKernel = PerformerKernel.exp,
        use_recurrent_decoding: bool = False,
        kernel_epsilon: float = 1e-4,
        normalize_output: bool = True,
        normalization_stabilizer: float = 1e-6,
        max_position_embeddings: int = 512,
        position_embedding_type: str | None = None,
        num_random_features: int | None = None,
        regularize_feature_norms: bool = True,
        causal: bool = False,
        feature_generation_algorithm: str
        | FeatureGenerationAlgorithm = (FeatureGenerationAlgorithm.qr),
        feature_redraw_interval: int | None = None,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        """
        Args:
        ----
        fields (`list[FieldInfo]`):
            A list of `FieldInfo` objects, each representing a field.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        kernel_type (`str` or `Callable` or `PerformerKernel`, *optional*, defaults to `PerformerKernel.exp`):
            The kernel to use for the attention layer. If a string, `"cosh"`, `"exp"`, `"elu"` and `"relu"` are supported.
        use_recurrent_decoding (`bool`, *optional*, defaults to `False`):
            Whether to use recurrent decoding or not.
        kernel_epsilon (`float`, *optional*, defaults to 1e-4):
            The epsilon to use for the kernel.
        normalize_output (`bool`, *optional*, defaults to `True`):
            Whether to normalize the output of the attention layer.
        normalization_stabilizer (`float`, *optional*, defaults to 1e-6):
            The stabilizer to use for normalization.
        use_linear_layers (`bool`, *optional*, defaults to `True`):
            Whether to use linear layers or not.
        linear_layer_names (`Sequence[str]`, *optional*, defaults to `("q_linear", "k_linear", "v_linear", "out_linear")`):
            The names of the linear layers.
        num_random_features (`int` or `None`, *optional*, defaults to `None`):
            The number of random features to use for the attention layer.
        use_thick_features (`bool`, *optional*, defaults to `False`):
            Whether to use thick features or not.
        regularize_feature_norms (`bool`, *optional*, defaults to `True`):
            Whether to regularize the feature norms or not.
        causal (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention from `fast_transformers` or not.
        orthogonal_feature_algorithm (`str` or `OrthogonalFeatureAlgorithm`, *optional*, defaults to `OrthogonalFeatureAlgorithm.auto`):
            The algorithm to use for orthogonal features. If a string, `"auto"`, `"kacs"` and `"qr"` are supported.
        feature_redraw_interval (`int` or `None`, *optional*, defaults to `100`):
            The interval at which to redraw the features. If `None`, the features are not redrawn.

        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.fields = fields
        self.label_columns = label_columns
        self.checkpoint = checkpoint
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.kernel_type = kernel_type
        self.use_recurrent_decoding = use_recurrent_decoding
        self.kernel_epsilon = kernel_epsilon
        self.normalize_output = normalize_output
        self.normalization_stabilizer = normalization_stabilizer
        self.num_random_features = num_random_features
        self.regularize_feature_norms = regularize_feature_norms
        self.causal = causal
        self.feature_generation_algorithm = feature_generation_algorithm
        self.feature_redraw_interval = feature_redraw_interval
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type


class SCNystromformerConfig(SCModelConfigBase):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    ----
        vocab_size (`int`, *optional*, defaults to 30000):
            Vocabulary size of the Nystromformer model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`NystromformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`NystromformerModel`].
        segment_means_seq_len (`int`, *optional*, defaults to 64):
            Sequence length used in segment-means.
        num_landmarks (`int`, *optional*, defaults to 64):
            The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
            matrix.
        conv_kernel_size (`int`, *optional*, defaults to 65):
            The kernel size of depthwise convolution used in Nystrom approximation.
        inverse_method (`str`, *optional*, defaults to `original`):
            Whether or not to use exact coefficient computation for the initial values for the iterative method of
            calculating the Moore-Penrose inverse of a matrix. Alternative iterative methods include newton and
            chebyshev.
        inverse_n_iter (`int`, *optional*, defaults to 6):
            Number of iterations to use to caculate the pseudoinverse of the matrix.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Example:
    -------
    ```python
    >>> from transformers import NystromformerModel, NystromformerConfig

    >>> # Initializing a Nystromformer uw-madison/nystromformer-512 style configuration
    >>> configuration = NystromformerConfig()

    >>> # Initializing a model from the uw-madison/nystromformer-512 style configuration
    >>> model = NystromformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "scnystromformer"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: torch.serialization.FILE_LIKE | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu_new",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: str | None = None,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        num_landmarks: int = 64,
        conv_kernel_size: int = 65,
        inverse_method: str = "original",
        inverse_n_iter: int = 6,
        pad_token_id: int = 2,
        use_cache: bool = True,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.fields = fields
        self.label_columns = label_columns
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inverse_method = inverse_method
        self.inverse_n_iter = inverse_n_iter
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.input_vocab_size = 0
        self.output_vocab_size = 0
        self.checkpoint = checkpoint
