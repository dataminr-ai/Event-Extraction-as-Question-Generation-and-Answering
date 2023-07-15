python src/train_qa_model.py  --lr 2e-5 \
                              --batch_size 8 --gpu 0 \
                              --gradient_accumulation_steps 4 \
                              --opt adam --multi_arg \
                              --transformer_name bart-large \
                              --run_id f4b1c879b0004c9b908fa73ff0c15767 --qg_model_type bart