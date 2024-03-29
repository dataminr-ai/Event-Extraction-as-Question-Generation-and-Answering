python src/train_question_generation.py --lr 3e-5 \
                                        --batch_size 8 --gpu 0 --opt 'adam'  \
                                        --transformer_name 'bart-large' \
                                        --models_dir  './model_checkpoint/qg_model_bart'