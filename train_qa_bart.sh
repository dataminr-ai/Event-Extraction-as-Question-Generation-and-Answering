python src/train_argument_extraction.py  --lr 2e-5 \
                              --batch_size 8 --gpu 0 \
                              --gradient_accumulation_steps 4 \
                              --opt adam --multi_arg \
                              --transformer_name bart-large \
                              --qg_model_type bart