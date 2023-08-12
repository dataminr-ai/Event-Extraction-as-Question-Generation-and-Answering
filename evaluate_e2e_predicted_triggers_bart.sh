python src/eval_with_predicted_trigger.py --batch_size 8 \
                        --gpu 0 --multi_arg \
                        --transformer_name bart-large \
                        --test_file ./model_checkpoint/trigger_model/trigger_predictions.json \
                        --qg_model_path ./model_checkpoint/qg_model_bart \
                        --eae_model_path ./model_checkpoint/eae_model_bart