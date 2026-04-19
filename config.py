config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 1e-5,
    'starting_epoch' : 0,
    'batch_size' : 1,
    'log_val' : 10,
    'task' : 'abnormal', # "meniscus" and  "acl" are the other options
    'weight_decay' : 0.01,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 32,
    'num_workers' : 4,
    'backbone_lr_mult' : 0.1,
    'clip_grad_norm' : 1.0
    ,
    'abnormal_prior_in_train' : True
}
