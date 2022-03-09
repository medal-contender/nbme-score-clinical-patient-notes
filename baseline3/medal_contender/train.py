from transformers import Trainer

from .utils import calculate_char_CV, get_location_predictions


class NBMETrainer(Trainer):
    """
    I override the default evaluation loop to include
    a character-level CV. The compute_metrics does the
    scoring at the token level, but the leaderboard score
    is at the character level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The Trainer hides some columns that are necessary for CV
        # This code makes those columns accessible
        dataset_type = kwargs["eval_dataset"].format["type"]
        dataset_columns = list(kwargs["eval_dataset"].features.keys())
        self.cv_dataset = kwargs["eval_dataset"].with_format(
            type=dataset_type, columns=dataset_columns)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):

        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

        # This same loop gets called during predict, and we can't do CV when predicting
        is_in_eval = metric_key_prefix == "eval"

        # Custom CV F1 calculation
        if is_in_eval:
            eval_preds = get_location_predictions(
                self.cv_dataset, eval_output.predictions)

            char_scores = calculate_char_CV(self.cv_dataset, eval_preds)

            for name, score in char_scores.items():
                eval_output.metrics[f"{metric_key_prefix}_char_{name}"] = score

        return eval_output
