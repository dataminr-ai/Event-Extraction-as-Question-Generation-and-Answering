python src/train_argument_extraction.py  --lr 20e-5 \
                              --batch_size 2 --gpu 0 \
                              --gradient_accumulation_steps 32 \
                              --opt 'adafactor' --multi_arg \
                              --transformer_name 't5-large' \
                              --qg_model_type 't5' \
                              --qg_model_path './model_checkpoint/qg_model_t5' \
                              --models_dir './model_checkpoint/eae_model_t5'
