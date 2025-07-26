from transformers import TrainingArguments, Trainer
import os
def get_training_args():
    return TrainingArguments(
        output_dir="../output_dir",
        auto_find_batch_size=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        optim="adamw_torch_fused",
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
def get_trainer(model, train_dataset, eval_dataset, compute_metrics=None,data_collator=None):
    training_args = get_training_args()
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
def get_lastest_checkpoint():
    last_checkpoint = None
    output_dir = "../output_dir"
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"Found checkpoint: {last_checkpoint}")
        else:
            print("No checkpoints found.")
    else:
        print("Output directory does not exist. Starting fresh.")
    return last_checkpoint