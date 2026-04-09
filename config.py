config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 3e-5,
    'starting_epoch' : 0,
    'batch_size' : 2,
    'log_val' : 10,
    'tasks' : ['abnormal', 'acl', 'meniscus'],
    'task_name' : 'multilabel',
    'weight_decay' : 3e-4,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 28,
    'num_workers' : 3
}
