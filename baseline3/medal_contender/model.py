import torch
from transformers.modeling_outputs import TokenClassifierOutput


def __init__(self, config):
    super(self.PreTrainedModel, self).__init__(config)

    kwargs = {
        "add_pooling_layer": False
    }
    if config.model_type not in {"bert", "roberta"}:
        kwargs = {}
    setattr(self, self.backbone_name, self.ModelClass(config, **kwargs))

    classifier_dropout_name = None
    for key in dir(config):
        if ("classifier" in key or "hidden" in key) and "dropout" in key:
            if getattr(config, key) is not None:
                classifier_dropout_name = key
                break

    if classifier_dropout_name is None:
        raise ValueError("Cannot infer dropout name in config")
    classifier_dropout = getattr(config, classifier_dropout_name)
    self.dropout = torch.nn.Dropout(classifier_dropout)
    self.classifier = torch.nn.Linear(config.hidden_size, 1)


def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Funky alert
    outputs = getattr(self, self.backbone_name)(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        # head_mask=head_mask, # these aren't necessary and some models error if you include
        # inputs_embeds=inputs_embeds,  # these aren't necessary and some models error if you include
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))

        # this ignores the part of the sequence that got -100 as labels
        loss = torch.masked_select(loss, labels.view(-1, 1) > -1).mean()

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@classmethod
def from_pretrained(cls, *args, **kwargs):
    cls.PreTrainedModel = kwargs.pop("PreTrainedModel")
    cls.ModelClass = kwargs.pop("ModelClass")
    cls.backbone_name = kwargs["config"].model_type

    # changes deberta-v2 --> deberta
    if "deberta" in cls.backbone_name:
        cls.backbone_name = "deberta"

    return super(cls.PreTrainedModel, cls).from_pretrained(*args, **kwargs)


def get_model(model_name_or_path, config):
    model_type = type(config).__name__[:-len("config")]
    name = f"{model_type}PreTrainedModel"
    PreTrainedModel = getattr(__import__("transformers", fromlist=[name]),
                              name)
    name = f"{model_type}Model"
    ModelClass = getattr(__import__("transformers", fromlist=[name]), name)

    model = type('HybridModel', (PreTrainedModel,), {
        '__init__': __init__,
        "forward": forward,
        "from_pretrained": from_pretrained
    })

    model._keys_to_ignore_on_load_unexpected = [r"pooler"]
    model._keys_to_ignore_on_load_missing = [r"position_ids"]

    return model.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        PreTrainedModel=PreTrainedModel,
        ModelClass=ModelClass,
        config=config
    )
